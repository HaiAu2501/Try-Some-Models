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
from langchain_community.tools import DuckDuckGoSearchRun
from experts import PROMPTS

load_dotenv()
warnings.filterwarnings("ignore")

llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))
# llm = ChatOpenAI(model="gemini-2.0-flash-exp", api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))
# llm = ChatOpenAI(model="gemini-exp-1206", api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))

class InputState(TypedDict):
    """
    Đầu vào của quy trình.

    Args:
    """
    pass

class OutputState(TypedDict):
    """
    Đầu ra của quy trình.

    Args:
    """
    pass

class OverallState(InputState, OutputState):
    """
    Lưu trữ trạng thái trung gian của quy trình.

    Args:
    """
    pass

web_search_tool = DuckDuckGoSearchRun()
res = web_search_tool.invoke("Python")

print(res)