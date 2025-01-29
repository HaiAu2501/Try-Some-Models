import os
import re
import pandas as pd
import numpy as np
import scipy

import warnings
from typing import TypedDict, List, Tuple, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.utilities import PythonREPL
from experts import PROMPTS

load_dotenv()
warnings.filterwarnings("ignore")

class Preprocess(BaseModel):
    code: str = Field(..., title = "Code", description = "Đoạn mã Python.")
    explanation: str = Field(..., title = "Explanation", description = "Giải thích về cách tiếp cận.")

class Review(BaseModel):
    should_continue: bool = Field(..., title = "Should Continue", description = "True nếu muốn tiếp tục cải thiện phương pháp, ngược lại False.")
    review: str = Field(..., title = "Review", description = "Phản hồi về phương pháp: đánh giá, góp ý, phê bình.")

llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))
# llm = ChatOpenAI(model="gemini-2.0-flash-exp", api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))
# llm = ChatOpenAI(model="gemini-exp-1206", api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))

preprocessor = llm.with_structured_output(Preprocess, strict=True)
critic = llm.with_structured_output(Review, strict=True)

class State(TypedDict):
    raw_data: pd.DataFrame                  # Dữ liệu thô
    processed_data: Dict[str, pd.DataFrame] # Dữ liệu sau khi xử lý
    cache: Dict[str, pd.DataFrame]          # Dữ liệu cache
    description: str                        # Mô tả dữ liệu thô
    review: Dict[str, str]                  # Đánh giá dữ liệu từ các tác tử
    goal: str                               # Mục tiêu bài toán
    should_continue: bool                   # True nếu muốn tiếp tục cải thiện phương pháp, ngược lại False
    current_step: str                       # Bước hiện tại
    cleaner_message: List[BaseMessage]      # Lịch sử trò chuyện với cleaner
    transformer_message: List[BaseMessage]  # Lịch sử trò chuyện với transformer
    extractor_message: List[BaseMessage]    # Lịch sử trò chuyện với extractor
    reviewer_message: List[BaseMessage]     # Lịch sử trò chuyện với reviewer
    max_attempts: int                       # Số lần thử tối đa
    
workflow: StateGraph = StateGraph(State)

def cleaner(state: State) -> Dict:
    max_attempts = 10
    attempts = 0

    if not state["cleaner_message"]:
        state["cleaner_message"] = [
            SystemMessage(content=PROMPTS["preprocess"]["cleaner"]),
            HumanMessage(content=state["description"])
        ]

    while attempts < max_attempts:
        attempts += 1
        response = preprocessor.invoke(state["cleaner_message"])
        code = response.code
        print(code)

        python_repl = PythonREPL(_globals={"df": state["raw_data"]})
        try:
            python_repl.run(code)
            cleaned_data = python_repl.globals["df"]
            state["processed_data"]["cleaned"] = cleaned_data.copy()
            state["cache"]["cleaned"] = cleaned_data
            state["review"]["cleaner"] = response.explanation
            message_content = f"Code của cleaner:\n{response.code}"
            state["cleaner_message"].append(HumanMessage(content=message_content))
            return state
        except Exception as e:
            print(e)
            continue
    return {"should_continue": False, "current_step": "cleaner"}

def transformer(state: State) -> Dict:
    max_attempts = 10
    attempts = 0

    if not state["transformer_message"]:
        state["transformer_message"] = [
            SystemMessage(content=PROMPTS["preprocess"]["transformer"]),
            HumanMessage(content=state["description"])
        ]

    while attempts < max_attempts:
        attempts += 1
        response = preprocessor.invoke(state["transformer_message"])
        code = response.code
        python_repl = PythonREPL(_globals={"df": state["cache"]["cleaned"]})
        try:
            python_repl.run(code)
            transformed_data = python_repl.globals["df"]
            state["processed_data"]["transformed"] = transformed_data.copy()
            state["cache"]["transformed"] = transformed_data
            state["review"]["transformer"] = response.explanation
            message_content = f"Code của transformer:\n{response.code}"
            state["transformer_message"].append(HumanMessage(content=message_content))
            return state
        except Exception as e:
            print(e)
            continue
    return {"should_continue": False, "current_step": "transformer"}

def extractor(state: State) -> Dict:
    max_attempts = 10
    attempts = 0

    if not state["extractor_message"]:
        state["extractor_message"] = [
            SystemMessage(content=PROMPTS["preprocess"]["extractor"]),
            HumanMessage(content=state["description"])
        ]

    while attempts < max_attempts:
        attempts += 1
        response = preprocessor.invoke(state["extractor_message"])
        code = response.code
        python_repl = PythonREPL(_globals={"df": state["processed_data"]["cleaned"]})
        try:
            python_repl.run(code)
            extracted_data = python_repl.globals["df"]
            state["processed_data"]["extracted"] = extracted_data.copy()
            state["cache"]["extracted"] = extracted_data
            state["review"]["extractor"] = response.explanation
            message_content = f"Code của extractor:\n{response.code}"
            state["extractor_message"].append(HumanMessage(content=message_content))
            return state
        except Exception as e:
            print(e)
            continue
    return {"should_continue": False, "current_step": "extractor"}

def reviewer(state: State) -> Dict:
    current_step = state["current_step"]
    
    if not state["reviewer_message"]:
        state["reviewer_message"] = [
            SystemMessage(content=PROMPTS["preprocess"]["reviewer"]),
            HumanMessage(content=state["description"])
        ]

    current_message = state[f"{current_step}_message"][-1]
    # state[f"{current_step}_message"].pop()
    state["reviewer_message"].append(current_message)
    response = critic.invoke(state["reviewer_message"])
    state["reviewer_message"].append(AIMessage(content=response.review))
    print(response.review)
    state[f"{current_step}_message"].append(HumanMessage(content=response.review))

    return {"should_continue": response.should_continue}

def router(state: State) -> str:
    """
    Xác định node tiếp theo dựa trên đánh giá từ reviewer.
    Args:
        state (State): Trạng thái hiện tại.
    Returns:
        str: Tên node tiếp theo ("cleaner", "transformer", "extractor" or "END").
    """
    if state["should_continue"]:
        return state["current_step"]
    if not state["should_continue"]:
        return {
            "cleaner": "transformer",
            "transformer": "extractor",
            "extractor": "END"
        }[state["current_step"]]

# Add nodes
workflow.add_node("cleaner", cleaner)
workflow.add_node("transformer", transformer)
workflow.add_node("extractor", extractor)
workflow.add_node("reviewer", reviewer)

# Add edges
workflow.add_edge(START, "cleaner")
workflow.add_edge("cleaner", "reviewer")
workflow.add_edge("transformer", "reviewer")
workflow.add_edge("extractor", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    router,
    {
        "cleaner": "cleaner",
        "transformer": "transformer",
        "extractor": "extractor",
        "END": END
    }
)

flow: CompiledStateGraph = workflow.compile()

initial_state: State = State(
    raw_data=pd.read_csv("data/data.csv", encoding="utf-8"),
    processed_data={},
    cache={},
    description="Dữ liệu là bảng chỉ gồm 3 cột: 'doanh số', 'lợi nhuận', 'chi phí'.",
    message=[],
    review={},
    goal="Chuẩn hóa dữ liệu.",
    should_continue=False,
    current_step="cleaner",
    cleaner_message=[],
    transformer_message=[],
    extractor_message=[],
    reviewer_message=[]
)

final_state = flow.invoke(initial_state)

print(final_state["processed_data"]["cleaned"])
print(final_state["processed_data"]["transformed"])
print(final_state["processed_data"]["extracted"])
