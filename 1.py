import os
import pandas as pd

import warnings
from typing import TypedDict, List, Dict, Literal, NotRequired
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_experimental.utilities import PythonREPL
from experts import PROMPTS

load_dotenv()
warnings.filterwarnings("ignore")

class Code(BaseModel):
    code: str = Field(..., title = "Code", description = "Đoạn mã Python.")
    explanation: str = Field(..., title = "Explanation", description = "Giải thích về cách tiếp cận.")

class Review(BaseModel):
    should_continue: bool = Field(..., title = "Should Continue", description = "True nếu muốn tiếp tục cải thiện phương pháp, ngược lại False.")
    review: str = Field(..., title = "Review", description = "Phản hồi về phương pháp: đánh giá, góp ý, phê bình.")

llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))
# llm = ChatOpenAI(model="gemini-2.0-flash-exp", api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))
# llm = ChatOpenAI(model="gemini-exp-1206", api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))

coder_llm = llm.with_structured_output(Code, strict=True)
reviewer_llm = llm.with_structured_output(Review, strict=True)

class State(TypedDict):
    """
    Lưu trữ trạng thái của workflow.

    Args:
        data_description: Mô tả về dữ liệu.
        goal: Mục tiêu của workflow.
        explanations: Giải thích về cách tiếp cận của cleaner và extractor.
        messages: Danh sách các tin nhắn giữa hệ thống và người dùng.
        cache: Dữ liệu cache.
        should_continue: True nếu muốn tiếp tục cải thiện phương pháp, ngược lại False.
        current_step: Bước hiện tại của workflow.
        attempt: Số lần thử.
        max_attempts: Số lần thử tối đa.
    """
    data_description: str                                
    goal: str
    messages: NotRequired[Dict[Literal["cleaner", "extractor", "reviewer"], List[BaseMessage]]]
    should_continue: NotRequired[bool]
    current_step: NotRequired[Literal["cleaner", "extractor", "END"]]
    attempt: NotRequired[int]
    max_attempts: NotRequired[int]                                                          

def initiator(state: State) -> Dict:
    """
    Tác tử initiator chịu trách nhiệm khởi tạo workflow.
    """
    messages: Dict[Literal["cleaner", "extractor", "reviewer"], List[BaseMessage]] = {
        "cleaner": [
            SystemMessage(content=PROMPTS["preprocess"]["cleaner"]),
        ],
        "extractor": [
            SystemMessage(content=PROMPTS["preprocess"]["extractor"])
        ],
        "reviewer": [
            SystemMessage(content=PROMPTS["preprocess"]["reviewer"])
        ],
    }     
    should_continue: bool = True            
    current_step: Literal["cleaner", "extractor", "END"] = "cleaner"
    attempt: int = 0    
    max_attempts: int = 3
    return {
        "messages": messages,
        "should_continue": should_continue,
        "current_step": current_step,
        "attempt": attempt,
        "max_attempts": max_attempts
    }

def cleaner(state: State) -> Dict:
    """
    Tác tử cleaner chịu trách nhiệm sinh code tiền xử lý dữ liệu thô.
    """
    messages = state["messages"]["cleaner"]
    messages.append(
        HumanMessage(
            content = (
                f"Mô tả dữ liệu: {state['data_description']}\n"
                f"Mục tiêu: {state['goal']}"
            )
        )
    )

    response = coder_llm.invoke(messages)
    messages.append(AIMessage(content=response.json()))
    state["messages"]["cleaner"] = messages
    return {
        "attempt": state["attempt"] + 1,
        "should_continue": True,
        "current_step": "cleaner",
    }

def extractor(state: State) -> Dict:
    """
    Tác tử extractor chịu trách nhiệm sinh code trích xuất thông tin từ dữ liệu đã được làm sạch.
    """
    messages = state["messages"]["extractor"]
    messages.append(
        HumanMessage(
            content = (
                f"Mô tả dữ liệu: {state['data_description']}\n"
                f"Mục tiêu: {state['goal']}"
            )
        )
    )

    response = coder_llm.invoke(messages)
    messages.append(AIMessage(content=response.json()))
    state["messages"]["extractor"] = messages
    return {
        "attempt": state["attempt"] + 1,
        "should_continue": True,
        "current_step": "extractor",
    }

def reviewer(state: State) -> Dict:
    """
    Tác tử reviewer chịu trách nhiệm đánh giá và đưa ra phản hồi về code và phương pháp.
    """
    if state["attempt"] >= state["max_attempts"]:
        return {
            "attempt": 0,
            "should_continue": False,
        }

    messages = state["messages"]["reviewer"]
    current_step = state["current_step"]
    coder_messages = state["messages"][current_step][-1]
    messages.append(coder_messages)
    messages.append(
        HumanMessage(
            content="Đưa ra chỉ thị cải tiến cho đoạn code trên."
        )
    )

    response = reviewer_llm.invoke(messages)
    messages.append(AIMessage(content=response.json()))
    state["messages"]["reviewer"] = messages
    return {
        "should_continue": response.should_continue,
    }

def router(state: State) -> Literal["cleaner", "extractor", "END"]:
    """
    Xác định node tiếp theo dựa trên đánh giá từ reviewer.
    Args:
        state (State): Trạng thái hiện tại.
    Returns:
        str: Tên node tiếp theo ("cleaner", "extractor" or "END").
    """
    if state["should_continue"]:
        return state["current_step"]
    if not state["should_continue"]:
        if state["current_step"] == "cleaner":
            return "extractor"
        if state["current_step"] == "extractor":
            return "END"

workflow: StateGraph = StateGraph(State)

# Add nodes
workflow.add_node("initiator", initiator)
workflow.add_node("cleaner", cleaner)
workflow.add_node("extractor", extractor)
workflow.add_node("reviewer", reviewer)

# Add edges
workflow.add_edge(START, "initiator")
workflow.add_edge("initiator", "cleaner")
workflow.add_edge("cleaner", "reviewer")
workflow.add_edge("extractor", "reviewer")
workflow.add_conditional_edges("reviewer", router, {"cleaner": "cleaner", "extractor": "extractor", "END": END})

flow: CompiledStateGraph = workflow.compile()

# initial_state: State = State(
#     data_description=(
#         "Dữ liệu gồm 5 cột:\n"
#         "- Country: Tên quốc gia, dạng chuỗi (string).\n"
#         "- GDP Growth Rate: Tỷ lệ tăng trưởng GDP, dạng số thực (float).\n"
#         "- Inflation Rate: Tỷ lệ lạm phát, dạng số thực (float).\n"
#         "- Unemployment Rate: Tỷ lệ thất nghiệp, dạng số thực (float).\n"
#         "- Consumer Price Index (CPI): Chỉ số giá tiêu dùng, dạng số thực (float)."
#     ),
#     goal="Phân tích dữ liệu để dự đoán tình hình kinh tế của các quốc gia.",
# )

# final_state: State = flow.invoke(initial_state)
