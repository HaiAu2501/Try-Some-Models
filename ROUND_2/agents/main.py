import os
from typing import List, Dict, Any, TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END


# Tải biến môi trường từ file .env (nơi chứa API key)
load_dotenv()

# Định nghĩa kiểu dữ liệu trạng thái
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_steps: List[str]


# Khởi tạo mô hình LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

# Định nghĩa các node trong đồ thị

# 1. Node xử lý tin nhắn của người dùng
def process_input(state: AgentState) -> AgentState:
    """Xử lý tin nhắn đầu vào và quyết định các bước tiếp theo."""
    # Lấy tin nhắn cuối cùng từ người dùng
    last_message = state["messages"][-1]
    
    # Tạo prompt để phân tích yêu cầu
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Bạn là một trợ lý AI thông minh. 
        Phân tích yêu cầu của người dùng và quyết định bước tiếp theo.
        Trả về MỘT trong các lựa chọn sau:
        - answer: nếu câu hỏi có thể trả lời trực tiếp
        - research: nếu cần thêm thông tin
        - clarify: nếu yêu cầu không rõ ràng
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="Phân tích yêu cầu và trả về bước tiếp theo.")
    ])
    
    # Chuẩn bị lịch sử chat trước đó (nếu có)
    chat_history = state["messages"][:-1] if len(state["messages"]) > 1 else []
    
    # Gọi LLM để phân tích
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"chat_history": chat_history})
    
    # Cập nhật trạng thái
    return {
        "messages": state["messages"],
        "next_steps": [result.strip()]
    }


# 2. Node trả lời trực tiếp
def answer_directly(state: AgentState) -> AgentState:
    """Cung cấp câu trả lời trực tiếp cho người dùng."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Bạn là một trợ lý AI hữu ích và thân thiện. Trả lời câu hỏi một cách đầy đủ và chính xác."),
        MessagesPlaceholder(variable_name="chat_history"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"chat_history": state["messages"]})
    
    # Thêm câu trả lời vào tin nhắn
    return {
        "messages": state["messages"] + [AIMessage(content=response)],
        "next_steps": []
    }


# 3. Node yêu cầu làm rõ
def ask_for_clarification(state: AgentState) -> AgentState:
    """Yêu cầu người dùng làm rõ yêu cầu."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Bạn là một trợ lý AI. Yêu cầu người dùng làm rõ thông tin một cách lịch sự."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="Tạo một câu hỏi để làm rõ yêu cầu của người dùng.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    clarification = chain.invoke({"chat_history": state["messages"]})
    
    return {
        "messages": state["messages"] + [AIMessage(content=clarification)],
        "next_steps": []
    }


# 4. Node nghiên cứu thêm
def research(state: AgentState) -> AgentState:
    """Giả lập việc tìm kiếm thêm thông tin."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Bạn là một trợ lý AI đang tìm kiếm thông tin. 
        Hãy giả lập việc nghiên cứu thêm và cung cấp thông tin chi tiết hơn.
        Nói rõ rằng đây là thông tin bạn đã tìm kiếm thêm."""),
        MessagesPlaceholder(variable_name="chat_history"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    research_result = chain.invoke({"chat_history": state["messages"]})
    
    return {
        "messages": state["messages"] + [AIMessage(content=research_result)],
        "next_steps": []
    }


# Định nghĩa hàm quyết định các bước tiếp theo
def decide_next_step(state: AgentState) -> str:
    next_step = state["next_steps"][0].lower() if state["next_steps"] else "answer"
    
    if "answer" in next_step:
        return "answer"
    elif "clarify" in next_step:
        return "clarify"
    elif "research" in next_step:
        return "research"
    else:
        return "answer"  # Mặc định trả lời trực tiếp


# Xây dựng đồ thị trạng thái
builder = StateGraph(AgentState)

# Thêm các node
builder.add_node("process", process_input)
builder.add_node("answer", answer_directly)
builder.add_node("clarify", ask_for_clarification)
builder.add_node("research", research)

# Định nghĩa luồng
builder.set_entry_point("process")
builder.add_conditional_edges(
    "process",
    decide_next_step,
    {
        "answer": "answer",
        "clarify": "clarify",
        "research": "research"
    }
)

# Đánh dấu các node kết thúc
builder.add_edge("answer", END)
builder.add_edge("clarify", END)
builder.add_edge("research", END)

# Khởi tạo đồ thị
graph = builder.compile()


# Hàm sử dụng tác tử để trả lời câu hỏi
def ask_agent(question: str, chat_history: List[BaseMessage] = None) -> List[BaseMessage]:
    if chat_history is None:
        chat_history = []
    
    # Tạo tin nhắn mới
    new_message = HumanMessage(content=question)
    
    # Thêm vào lịch sử chat
    messages = chat_history + [new_message]
    
    # Khởi tạo trạng thái ban đầu
    initial_state = {
        "messages": messages,
        "next_steps": []
    }
    
    # Chạy đồ thị
    result = graph.invoke(initial_state)
    
    # Trả về tin nhắn sau khi xử lý
    return result["messages"]


# Ví dụ sử dụng
if __name__ == "__main__":
    # Tương tác đơn
    print("=== Tương tác đơn ===")
    messages = ask_agent("Tìm hiểu về việc học máy là gì?")
    for message in messages:
        print(f"{message.type}: {message.content}\n")
    
    # Tương tác đa vòng
    print("\n=== Tương tác đa vòng ===")
    chat_history = []
    
    # Vòng 1
    messages = ask_agent("Bạn có thể giải thích về cách hoạt động của mô hình GPT không?", chat_history)
    chat_history = messages
    for message in messages:
        print(f"{message.type}: {message.content}\n")
    
    # Vòng 2
    messages = ask_agent("Vậy còn so sánh với BERT thì sao?", chat_history)
    for message in messages[-1:]:  # Chỉ in tin nhắn mới
        print(f"{message.type}: {message.content}\n")