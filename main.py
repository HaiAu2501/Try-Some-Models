import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List, Tuple
from pydantic import BaseModel, Field

from experts import PROMPTS

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

test = "Dữ liệu là 1 file csv gồm các cột: 'doanh số', 'lợi nhuận', 'chi phí'."

message: List[Tuple[str, str]] = [
    ("system", PROMPTS["preprocess"]["extractor"]),
    ("human", test)
]

class Preprocess(BaseModel):
    code: str = Field(..., title="Code", description="The code for cleaning the data.")
    explanation: str = Field(..., title="Explanation", description="The explanation of the process.")
    input: str = Field(..., title="Input", description="The description of the input data.")
    output: str = Field(..., title="Output", description="The description of the output data.")

bot = llm.with_structured_output(Preprocess, strict=True)

response = bot.invoke(message)

print(response.code)
print(response.explanation)
print(response.input)
print(response.output)