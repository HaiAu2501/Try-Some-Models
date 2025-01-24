
import os




from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from typing import TypedDict, List, Tuple, Dict
from pydantic import BaseModel, Field
import pandas as pd
import re
from experts import PROMPTS

class State(TypedDict):
    raw_data: pd.DataFrame         # Dữ liệu thô
    processed_data: pd.DataFrame   # Dữ liệu đã qua xử lý
    description: str               # Mô tả dữ liệu thô
    message: List[Tuple[str, str]] # Lịch sử trò chuyện
    review: Dict[str, str]         # Đánh giá dữ liệu từ các tác tử
    goal: str                      # Mục tiêu bài toán

class Preprocess(BaseModel):
    code: str = Field(..., title = "Code", description = "The code for cleaning the data.")
    explanation: str = Field(..., title = "Explanation", description = "The explanation of the process.")
    input: str = Field(..., title = "Input", description = "The description of the input data.")
    output: str = Field(..., title = "Output", description = "The description of the output data.")

openai_llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))
preprocessor = openai_llm.with_structured_output(Preprocess, strict = True)

workflow = StateGraph(State)

def cleaner(state: State) -> State:
    # Placeholder
    message = {
        ("system", PROMPTS["preprocess"]["cleaner"]),
        ("human", state["description"])
    }
    response = preprocessor.invoke(message)
    code = response.code

    df = state["raw_data"]


    return state

def transformer(state: State) -> State:
    # Placeholder
    return state

def extractor(state: State) -> State:
    # Placeholder
    return state

workflow.set_entry_point("cleaner")
workflow.add_node("cleaner", cleaner)
workflow.add_edge("cleaner", END)
