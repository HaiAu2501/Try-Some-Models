import os
import pandas as pd
import warnings
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from experts import PROMPTS

load_dotenv()
warnings.filterwarnings("ignore")

llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))
# llm = ChatOpenAI(model="gemini-2.0-flash-exp", api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))
# llm = ChatOpenAI(model="gemini-exp-1206", api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("GEMINI_BASE_URL"))

# -------------------------------------------
# Định nghĩa các model dữ liệu cho LLM
# -------------------------------------------
class Code(BaseModel):
    code: str = Field(..., title="Code", description="Đoạn mã Python được sinh ra.")
    explanation: str = Field(..., title="Explanation", description="Giải thích về cách tiếp cận.")

class Review(BaseModel):
    should_continue: bool = Field(
        ..., 
        title="Should Continue", 
        description="True nếu cần tiếp tục cải tiến, False nếu dừng lại."
    )
    review: str = Field(..., title="Review", description="Phản hồi từ Reviewer.")

coder = llm.with_structured_output(Code, strict=True)
reviewer = llm.with_structured_output(Review, strict=True)

class Configurations(TypedDict):
    data_description: str
    goal: str


class CleanerAgent:
    def __init__(self, llm, system_prompt: str, config: Configurations):
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config

        self.history: List[BaseMessage] = []

    def run(self):
        if not self.history:
            self.history.append(SystemMessage(content=self.system_prompt))
            input = self.config["data_description"] + "\n" + self.config["goal"]
            self.history.append(HumanMessage(content=input))

        

    
    pass

# -------------------------------------------
# Chạy Pipeline
# -------------------------------------------
if __name__ == "__main__":
    # Giả sử dữ liệu được đọc từ file CSV
    df = pd.read_csv("data/data.csv")
    description = "Dữ liệu bao gồm các cột: 'doanh số', 'lợi nhuận', 'chi phí'."
    goal = "Tiền xử lý dữ liệu và tạo ra đặc trưng mới để sử dụng trong mô hình dự báo."
    
    
    print("Code cuối cùng của Cleaner:")
    print("Code cuối cùng của Extractor:")
