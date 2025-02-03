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

class InputState(TypedDict):
    """
    Đầu vào của quy trình.

    Args:
        data_description: Mô tả về dữ liệu.
        goal: Mục tiêu của quy trình.
    """
    data_description: str
    goal: str

class OutputState(TypedDict):
    """
    Đầu ra của quy trình.

    Args:
        answers: Đoạn mã Python và giải thích của cleaner và extractor.
    """
    answers: Dict[Literal["cleaner", "extractor"], Code]

class OverallState(InputState, OutputState):
    """
    Lưu trữ trạng thái trung gian của quy trình.

    Args:
        messages: Danh sách các tin nhắn giữa hệ thống và người dùng.
        should_continue: True nếu muốn tiếp tục cải thiện phương pháp, ngược lại False.
        current_step: Bước hiện tại của workflow.
        attempt: Số lần thử.
        max_attempts: Số lần thử tối đa.
    """
    # Process
    messages: Dict[Literal["cleaner", "extractor", "reviewer"], List[BaseMessage]]
    should_continue: bool
    current_step: Literal["cleaner", "extractor", "END"]
    attempt: int
    max_attempts: int                               

def initiator(state: OverallState) -> Dict:
    """
    Tác tử initiator chịu trách nhiệm khởi tạo workflow.
    """
    state["messages"] = {
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
    state["should_continue"] = True          
    state["current_step"] = "cleaner"
    state["attempt"] = 0
    state["max_attempts"] = 3
    state["answers"] = {}
    return state

def cleaner(state: OverallState) -> Dict:
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
    state["answers"]["cleaner"] = response
    state["messages"]["cleaner"] = messages
    return {
        "attempt": state["attempt"] + 1,
        "should_continue": True,
        "current_step": "cleaner",
    }

def extractor(state: OverallState) -> Dict:
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
    state["answers"]["extractor"] = response
    state["messages"]["extractor"] = messages
    return {
        "attempt": state["attempt"] + 1,
        "should_continue": True,
        "current_step": "extractor",
    }

def reviewer(state: OverallState) -> Dict:
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

def router(state: OverallState) -> Literal["cleaner", "extractor", "END"]:
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

workflow: StateGraph = StateGraph(OverallState, input=InputState, output=OutputState)

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

graph: CompiledStateGraph = workflow.compile()

# initial_state: InputState = InputState(
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

# final_state: OutputState = graph.invoke(initial_state)
