import os
import pandas as pd

import warnings
from typing import TypedDict, List, Dict, Literal
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
    """
    Lưu trữ trạng thái của workflow.

    Args:
        data_description: Mô tả về dữ liệu.
        goal: Mục tiêu của workflow.
        explanations: Giải thích về cách tiếp cận của cleaner và extractor.
        messages: Danh sách các tin nhắn giữa hệ thống và người dùng.
        should_continue: True nếu muốn tiếp tục cải thiện phương pháp, ngược lại False.
        current_step: Bước hiện tại của workflow.
        attempt: Số lần thử.
        max_attempts: Số lần thử tối đa.
    """
    data_description: str                                
    goal: str                                            
    explanations: Dict[Literal["cleaner", "extractor"], str]                  
    messages: Dict[Literal["cleaner", "extractor", "reviewer"], List[BaseMessage]]             
    should_continue: bool            
    current_step: Literal["cleaner", "extractor", "END"] 
    attempt: int                                       
    max_attempts: int

def cleaner(state: State) -> Dict:
    return {}

def extractor(state: State) -> Dict:
    return {}

def reviewer(state: State) -> Dict:
    return {}

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
            "cleaner": "extractor",
            "extractor": "END"
        }[state["current_step"]]

workflow: StateGraph = StateGraph(State)

# Add nodes
workflow.add_node("cleaner", cleaner)
workflow.add_node("extractor", extractor)
workflow.add_node("reviewer", reviewer)

# Add edges
workflow.add_edge(START, "cleaner")
workflow.add_edge("cleaner", "reviewer")
workflow.add_edge("extractor", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    router,
    {
        "cleaner": "cleaner",
        "extractor": "extractor",
        "END": END
    }
)

flow: CompiledStateGraph = workflow.compile()

initial_state: State = State(
    data_description="Dữ liệu là một bảng dữ liệu với các cột và hàng.",
    goal="Tách dữ liệu thành các cột và hàng.",
    explanations={},
    messages={
        "cleaner": [
            SystemMessage(content=PROMPTS["preprocess"]["cleaner"])
        ],
        "extractor": [
            SystemMessage(content=PROMPTS["preprocess"]["extractor"])
        ],
        "reviewer": [
            SystemMessage(content=PROMPTS["preprocess"]["reviewer"])
        ],
    },
    should_continue=True,
    current_step="cleaner",
    attempt=0,
    max_attempts=3
)

final_state = flow.invoke(initial_state)
