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
    code: str = Field(..., title = "Code", description = "The Python code.")
    explanation: str = Field(..., title = "Explanation", description = "The explanation of the process.")

# openai_llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))
# preprocessor = openai_llm.with_structured_output(Preprocess, strict = True)

gemini = ChatOpenAI(model="gemini-exp-1206", api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))
preprocessor = gemini.with_structured_output(Preprocess, strict=True)

class State(TypedDict):
    raw_data: pd.DataFrame                  # Dữ liệu thô
    processed_data: Dict[str, pd.DataFrame] # Dữ liệu sau khi xử lý
    cache: Dict[str, pd.DataFrame]          # Dữ liệu cache
    description: str                        # Mô tả dữ liệu thô
    message: List[Tuple[str, str]]          # Lịch sử trò chuyện
    review: Dict[str, str]                  # Đánh giá dữ liệu từ các tác tử
    goal: str                               # Mục tiêu bài toán

workflow: StateGraph = StateGraph(State)

def cleaner(state: State) -> State:
    max_attempts = 50
    attempts = 0

    message = [
        SystemMessage(content=PROMPTS["preprocess"]["cleaner"]),
        HumanMessage(content=state["description"])
    ]

    while attempts < max_attempts:
        attempts += 1
        response = preprocessor.invoke(message)
        code = response.code

        python_repl = PythonREPL(_globals={"df": state["raw_data"]})
        try:
            python_repl.run(code)
            cleaned_data = python_repl.globals["df"]
            state["processed_data"]["cleaned"] = cleaned_data.copy()
            state["cache"]["cleaned"] = cleaned_data
            state["review"]["cleaner"] = response.explanation
            return state
        except Exception as e:
            print(e)
            continue
    return state

def transformer(state: State) -> State:
    max_attempts = 50
    attempts = 0

    message = [
        SystemMessage(content=PROMPTS["preprocess"]["transformer"]),
        HumanMessage(content=state["description"])
    ]

    while attempts < max_attempts:
        attempts += 1
        response = preprocessor.invoke(message)
        code = response.code

        python_repl = PythonREPL(_globals={"df": state["cache"]["cleaned"]})
        try:
            python_repl.run(code)
            transformed_data = python_repl.globals["df"]
            state["processed_data"]["transformed"] = transformed_data.copy()
            state["cache"]["transformed"] = transformed_data
            state["review"]["transformer"] = response.explanation
            return state
        except Exception as e:
            print(e)
            continue
    return state

def extractor(state: State) -> State:
    max_attempts = 50
    attempts = 0

    message = [
        SystemMessage(content=PROMPTS["preprocess"]["extractor"]),
        HumanMessage(content=state["description"])
    ]

    while attempts < max_attempts:
        attempts += 1
        response = preprocessor.invoke(message)
        code = response.code

        python_repl = PythonREPL(_globals={"df": state["cache"]["transformed"]})
        try:
            python_repl.run(code)
            extracted_data = python_repl.globals["df"]
            state["processed_data"]["extracted"] = extracted_data.copy()
            state["cache"]["extracted"] = extracted_data
            state["review"]["extractor"] = response.explanation
            return state
        except Exception as e:
            print(e)
            continue
    return state

# Add nodes
workflow.add_node("cleaner", cleaner)
workflow.add_node("transformer", transformer)
workflow.add_node("extractor", extractor)

# Add edges
workflow.add_edge(START, "cleaner")
workflow.add_edge("cleaner", "transformer")
workflow.add_edge("transformer", "extractor")
workflow.add_edge("extractor", END)

flow: CompiledStateGraph = workflow.compile()

initial_state: State = State(
	raw_data = pd.read_csv("data/data.csv", encoding="utf-8"),
	processed_data = {},
    cache = {},
	description = "Dữ liệu là bảng gồm 3 cột: 'doanh số', 'lợi nhuận', 'chi phí'.",
	message = [],
	review = {},
	goal = "Chuẩn hóa dữ liệu."
)

final_state = flow.invoke(initial_state)

print(final_state["processed_data"]["cleaned"])
print(final_state["processed_data"]["transformed"])
print(final_state["processed_data"]["extracted"])
