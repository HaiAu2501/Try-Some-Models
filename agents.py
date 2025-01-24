from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
from experts import PROMPTS



load_dotenv()

openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

cleaner_prompt = SystemMessage(PROMPTS["preprocess"]["cleaner"])
transformer_prompt = SystemMessage(PROMPTS["preprocess"]["transformer"])
extractor_prompt = SystemMessage(PROMPTS["preprocess"]["extractor"])

cleaner_agent = create_react_agent(openai_llm, cleaner_prompt)

