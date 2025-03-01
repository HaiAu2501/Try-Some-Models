from typing import Dict, List, Any, Annotated, TypedDict, cast
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document
from langchain_openai import ChatOpenAI

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Now import with relative paths
from experts.group_1 import (
    ECONOMETRICIAN, EMPIRICAL_ECONOMIST, NORMATIVE_ECONOMIST, 
    MACROECONOMIST, MICROECONOMIST
)
from experts.group_2 import (
    BEHAVIORAL_ECONOMIST, SOCIO_ECONOMIST
)
from experts.group_3 import (
    CORPORATE_MANAGEMENT_EXPERT, FINANCIAL_ECONOMIST, INTERNATIONAL_ECONOMIST,
    LOGISTICS_AND_SUPPLY_CHAIN_EXPERT, TRADE_AND_COMMERCE_EXPERT
)
from experts.group_4 import (
    DIGITAL_ECONOMY_AND_INNOVATION_EXPERT, ENVIRONMENTAL_ECONOMIST,
    PUBLIC_POLICY_AND_POLITICAL_ECONOMY_EXPERT
)

# Load environment variables
load_dotenv()

# Define state structures
class InputState(TypedDict):
    input_data: str                  # Data to be analyzed as string
    file_name: str                   # Name of the file being analyzed

class OutputState(TypedDict):
    analyses: Annotated[Dict[str, str], "merge"]  # Analyses from different agents - merge each agent's analysis
    group_summaries: Annotated[Dict[str, str], "merge"]  # Summaries from each group - merge each group's summary
    final_report: str                # Final combined report

class AgentState(InputState, OutputState):
    """Combined state for the agent system, inheriting both input and output states."""
    pass

# Function to create a model based on environment configuration
def get_model():
    """
    Get the appropriate LLM model based on environment configuration.
    Returns OpenAI model by default.
    """
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("No API key found for OpenAI")

# Generic expert agent creation function
def create_expert_agent(system_prompt: str, agent_name: str):
    """Create an expert agent with the given system prompt and name."""
    model = get_model()
    
    # Define the function for this agent node
    def expert_analysis(state: AgentState) -> AgentState:
        """Run expert analysis on input data and store in state."""
        try:
            # Trích xuất giá trị từ state
            input_data = state.get("input_data", "")
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running {agent_name} on file: {file_name}")
            
            # Tạo messages trực tiếp
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Phân tích dữ liệu sau đây dựa theo chuyên môn của bạn:
                
                {input_data}
                
                Tài liệu này có tên: {file_name}
                
                Hãy đưa ra nhận định, đánh giá và khuyến nghị chi tiết liên quan đến lĩnh vực chuyên môn của bạn.
                Tập trung vào các yếu tố quan trọng nhất mà bạn phát hiện từ dữ liệu.
                """)
            ]
            
            # Gọi model
            response = model.invoke(messages)
            analysis = response.content
            
        except Exception as e:
            print(f"[ERROR] Error in {agent_name}: {str(e)}")
            analysis = f"Error analyzing with {agent_name}: {str(e)}"
        
        # Create a new state with only the analysis for this expert
        new_state = cast(AgentState, {})
        new_state["analyses"] = {agent_name: analysis}
        
        return new_state
    
    return expert_analysis
    
    return expert_analysis

# Create expert agent nodes for each group

# Group 1: Academic Quantitative
econometrician_node = create_expert_agent(ECONOMETRICIAN, "econometrician")
empirical_economist_node = create_expert_agent(EMPIRICAL_ECONOMIST, "empirical_economist")
normative_economist_node = create_expert_agent(NORMATIVE_ECONOMIST, "normative_economist")
macroeconomist_node = create_expert_agent(MACROECONOMIST, "macroeconomist")
microeconomist_node = create_expert_agent(MICROECONOMIST, "microeconomist")

# Group 2: Behavioral Social
behavioral_economist_node = create_expert_agent(BEHAVIORAL_ECONOMIST, "behavioral_economist")
socio_economist_node = create_expert_agent(SOCIO_ECONOMIST, "socio_economist")

# Group 3: Market Business
corporate_management_node = create_expert_agent(CORPORATE_MANAGEMENT_EXPERT, "corporate_management")
financial_economist_node = create_expert_agent(FINANCIAL_ECONOMIST, "financial_economist")
international_economist_node = create_expert_agent(INTERNATIONAL_ECONOMIST, "international_economist")
logistics_node = create_expert_agent(LOGISTICS_AND_SUPPLY_CHAIN_EXPERT, "logistics_expert")
trade_commerce_node = create_expert_agent(TRADE_AND_COMMERCE_EXPERT, "trade_commerce_expert")

# Group 4: Policy Innovation
digital_economy_node = create_expert_agent(DIGITAL_ECONOMY_AND_INNOVATION_EXPERT, "digital_economy_expert")
environmental_economist_node = create_expert_agent(ENVIRONMENTAL_ECONOMIST, "environmental_economist")
public_policy_node = create_expert_agent(PUBLIC_POLICY_AND_POLITICAL_ECONOMY_EXPERT, "public_policy_expert")

# Create group summarizer nodes
def create_group_summarizer(group_name: str, expert_names: List[str]):
    """Create a summarizer for a group of experts."""
    model = get_model()
    
    def summarize_group(state: AgentState) -> AgentState:
        """Summarize analyses from experts in this group."""
        try:
            # Extract analyses from this group's experts
            expert_analyses = ""
            for expert in expert_names:
                if expert in state.get("analyses", {}):
                    expert_analyses += f"### Phân tích từ {expert}:\n{state['analyses'][expert]}\n\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running summarizer for {group_name} on file: {file_name}")
            
            # Tạo messages trực tiếp
            messages = [
                SystemMessage(content=f"""
                Bạn là người tổng hợp ý kiến cho nhóm chuyên gia {group_name}.
                Nhiệm vụ của bạn là tổng hợp, kết nối và đúc kết các ý kiến từ các chuyên gia trong nhóm.
                Hãy tìm ra điểm chung, điểm khác biệt và đưa ra kết luận tổng thể từ góc nhìn của nhóm.
                """),
                HumanMessage(content=f"""
                Tổng hợp các phân tích sau đây từ các chuyên gia trong nhóm:
                
                {expert_analyses}
                
                Tài liệu đang phân tích có tên: {file_name}
                
                Hãy đưa ra tổng kết toàn diện về góc nhìn của nhóm chuyên gia này.
                """)
            ]
            
            # Gọi model
            response = model.invoke(messages)
            summary = response.content
            
        except Exception as e:
            print(f"[ERROR] Error in {group_name} summarizer: {str(e)}")
            summary = f"Error summarizing {group_name}: {str(e)}"
        
        # Create a new state with only the group summary
        new_state = cast(AgentState, {})
        new_state["group_summaries"] = {group_name: summary}
        
        return new_state
    
    return summarize_group
    
    return summarize_group

# Create the final report synthesizer
def create_final_synthesizer():
    """Create the final synthesizing node that combines all group summaries."""
    model = get_model()
    
    def synthesize_final_report(state: AgentState) -> AgentState:
        """Create the final synthesized report from all group summaries."""
        try:
            # Format all group summaries
            group_summaries_text = ""
            for group_name, summary in state.get("group_summaries", {}).items():
                group_summaries_text += f"### Tổng kết từ nhóm {group_name}:\n{summary}\n\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running final synthesizer on file: {file_name}")
            
            # Tạo messages trực tiếp
            messages = [
                SystemMessage(content="""
                Bạn là chuyên gia tổng hợp báo cáo cuối cùng.
                Nhiệm vụ của bạn là kết hợp tất cả các tổng kết từ các nhóm chuyên gia khác nhau 
                thành một báo cáo toàn diện, mạch lạc và có tính ứng dụng cao.
                
                Hãy đưa ra các kết luận, khuyến nghị cụ thể và định hướng hành động.
                """),
                HumanMessage(content=f"""
                Tổng hợp các phân tích từ các nhóm chuyên gia sau:
                
                {group_summaries_text}
                
                Tài liệu đang phân tích có tên: {file_name}
                
                Hãy tạo báo cáo cuối cùng với cấu trúc:
                1. Tóm tắt tổng quan
                2. Phát hiện chính từ mỗi nhóm chuyên gia
                3. Mối liên hệ và tính nhất quán giữa các phân tích
                4. Khuyến nghị và hướng đi cụ thể
                5. Kết luận
                """)
            ]
            
            # Gọi model
            response = model.invoke(messages)
            final_report = response.content
            
        except Exception as e:
            print(f"[ERROR] Error in final synthesizer: {str(e)}")
            final_report = f"Error generating final report: {str(e)}"
        
        # Create a new state with only the final report
        new_state = cast(AgentState, {})
        new_state["final_report"] = final_report
        
        return new_state
    
    return synthesize_final_report
    
    return synthesize_final_report

# Create the group summarizers
academic_group_summarizer = create_group_summarizer(
    "Học Thuật Định Lượng (Academic Quantitative)",
    ["econometrician", "empirical_economist", "normative_economist", "macroeconomist", "microeconomist"]
)

behavioral_group_summarizer = create_group_summarizer(
    "Hành Vi Xã Hội (Behavioral Social)",
    ["behavioral_economist", "socio_economist"]
)

market_group_summarizer = create_group_summarizer(
    "Thị Trường Doanh Nghiệp (Market Business)",
    ["corporate_management", "financial_economist", "international_economist", 
     "logistics_expert", "trade_commerce_expert"]
)

policy_group_summarizer = create_group_summarizer(
    "Chính Sách Đổi Mới (Policy Innovation)",
    ["digital_economy_expert", "environmental_economist", "public_policy_expert"]
)

# Create the final synthesizer node
final_synthesizer = create_final_synthesizer()