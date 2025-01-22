import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

message = [
    ("system", "You are a chatbot."),
    ("human", "What is your name?"),
]

response = llm.invoke(message)

print(response.content)