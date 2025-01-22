import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List, Tuple
from experts import PROMPTS
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

test = "Dữ liệu là 1 file csv gồm các cột: 'doanh số', 'lợi nhuận', 'chi phí'."

message: List[Tuple[str, str]] = [
    ("system", PROMPTS["preprocess"]["cleaner"]),
    ("human", test)
]

class Cleaner(BaseModel):
    explanation: str = Field(..., title="Explanation", description="The explanation of the cleaning process.")
    code: str = Field(..., title="Code", description="The code for cleaning the data.")

bot = llm.with_structured_output(Cleaner, strict=True)

response = bot.invoke(message)

print(response.explanation)
print(response.code)