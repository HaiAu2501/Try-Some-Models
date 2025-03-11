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
    MARKET_ANALYST, TECHNICAL_ANALYST, FUNDAMENTAL_ANALYST, 
    SENTIMENT_ANALYST, ECONOMIC_INDICATORS_EXPERT
)
from experts.group_2 import (
    FINANCIAL_STATEMENT_ANALYST, FINANCIAL_RATIO_EXPERT, VALUATION_EXPERT,
    CASH_FLOW_ANALYST, CAPITAL_STRUCTURE_EXPERT
)
from experts.group_3 import (
    BANKING_FINANCE_EXPERT, REAL_ESTATE_EXPERT, CONSUMER_GOODS_EXPERT,
    INDUSTRIAL_EXPERT, TECHNOLOGY_EXPERT
)
from experts.group_4 import (
    GLOBAL_MARKETS_EXPERT, GEOPOLITICAL_RISK_ANALYST, REGULATORY_FRAMEWORK_EXPERT,
    MONETARY_POLICY_EXPERT, DEMOGRAPHIC_TRENDS_EXPERT
)
from experts.group_5 import (
    GAME_THEORY_STRATEGIST, RISK_MANAGEMENT_EXPERT, PORTFOLIO_OPTIMIZATION_EXPERT,
    ASSET_ALLOCATION_STRATEGIST, INVESTMENT_PSYCHOLOGY_EXPERT
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
                Đặc biệt, hãy đề cập đến ý nghĩa của các phát hiện đối với chiến lược đầu tư trên thị trường Việt Nam.
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

# Create expert agent nodes for each group

# Group 1: Market Analysis
market_analyst_node = create_expert_agent(MARKET_ANALYST, "market_analyst")
technical_analyst_node = create_expert_agent(TECHNICAL_ANALYST, "technical_analyst")
fundamental_analyst_node = create_expert_agent(FUNDAMENTAL_ANALYST, "fundamental_analyst")
sentiment_analyst_node = create_expert_agent(SENTIMENT_ANALYST, "sentiment_analyst")
economic_indicators_node = create_expert_agent(ECONOMIC_INDICATORS_EXPERT, "economic_indicators_expert")

# Group 2: Financial Analysis
financial_statement_node = create_expert_agent(FINANCIAL_STATEMENT_ANALYST, "financial_statement_analyst")
financial_ratio_node = create_expert_agent(FINANCIAL_RATIO_EXPERT, "financial_ratio_expert")
valuation_node = create_expert_agent(VALUATION_EXPERT, "valuation_expert")
cash_flow_node = create_expert_agent(CASH_FLOW_ANALYST, "cash_flow_analyst")
capital_structure_node = create_expert_agent(CAPITAL_STRUCTURE_EXPERT, "capital_structure_expert")

# Group 3: Sectoral Analysis
banking_finance_node = create_expert_agent(BANKING_FINANCE_EXPERT, "banking_finance_expert")
real_estate_node = create_expert_agent(REAL_ESTATE_EXPERT, "real_estate_expert")
consumer_goods_node = create_expert_agent(CONSUMER_GOODS_EXPERT, "consumer_goods_expert")
industrial_node = create_expert_agent(INDUSTRIAL_EXPERT, "industrial_expert")
technology_node = create_expert_agent(TECHNOLOGY_EXPERT, "technology_expert")

# Group 4: External Factors
global_markets_node = create_expert_agent(GLOBAL_MARKETS_EXPERT, "global_markets_expert")
geopolitical_risk_node = create_expert_agent(GEOPOLITICAL_RISK_ANALYST, "geopolitical_risk_analyst")
regulatory_framework_node = create_expert_agent(REGULATORY_FRAMEWORK_EXPERT, "regulatory_framework_expert")
monetary_policy_node = create_expert_agent(MONETARY_POLICY_EXPERT, "monetary_policy_expert")
demographic_trends_node = create_expert_agent(DEMOGRAPHIC_TRENDS_EXPERT, "demographic_trends_expert")

# Group 5: Strategy
game_theory_node = create_expert_agent(GAME_THEORY_STRATEGIST, "game_theory_strategist")
risk_management_node = create_expert_agent(RISK_MANAGEMENT_EXPERT, "risk_management_expert")
portfolio_optimization_node = create_expert_agent(PORTFOLIO_OPTIMIZATION_EXPERT, "portfolio_optimization_expert")
asset_allocation_node = create_expert_agent(ASSET_ALLOCATION_STRATEGIST, "asset_allocation_strategist")
investment_psychology_node = create_expert_agent(INVESTMENT_PSYCHOLOGY_EXPERT, "investment_psychology_expert")

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
                Tập trung vào ý nghĩa của các phân tích đối với chiến lược đầu tư trên thị trường Việt Nam.
                """),
                HumanMessage(content=f"""
                Tổng hợp các phân tích sau đây từ các chuyên gia trong nhóm:
                
                {expert_analyses}
                
                Tài liệu đang phân tích có tên: {file_name}
                
                Hãy đưa ra tổng kết toàn diện về góc nhìn của nhóm chuyên gia này, tập trung vào các khuyến nghị đầu tư cụ thể.
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
                Bạn là chuyên gia tổng hợp báo cáo cuối cùng về chiến lược đầu tư.
                Nhiệm vụ của bạn là kết hợp tất cả các tổng kết từ các nhóm chuyên gia khác nhau 
                thành một báo cáo toàn diện về chiến lược đầu tư tối ưu cho thị trường Việt Nam.
                
                Hãy đưa ra các kết luận, chiến lược đầu tư cụ thể và kế hoạch hành động chi tiết.
                Tập trung vào việc xây dựng chiến lược đầu tư có thể thực hiện được, với các khuyến nghị
                về phân bổ tài sản, lựa chọn cổ phiếu/ngành, thời điểm mua/bán và quản lý rủi ro.
                """),
                HumanMessage(content=f"""
                Tổng hợp các phân tích từ các nhóm chuyên gia sau:
                
                {group_summaries_text}
                
                Tài liệu đang phân tích có tên: {file_name}
                
                Hãy tạo báo cáo chiến lược đầu tư cuối cùng với cấu trúc:
                1. Tóm tắt tổng quan về thị trường
                2. Phân tích cơ hội và rủi ro đầu tư
                3. Chiến lược đầu tư tối ưu
                   a. Phân bổ tài sản chiến lược
                   b. Lựa chọn ngành và cổ phiếu tiềm năng
                   c. Thời điểm tham gia thị trường
                4. Kế hoạch quản lý rủi ro
                5. Kết luận và khuyến nghị hành động cụ thể
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

# Create the group summarizers
market_analysis_group_summarizer = create_group_summarizer(
    "Phân tích Thị trường (Market Analysis)",
    ["market_analyst", "technical_analyst", "fundamental_analyst", 
     "sentiment_analyst", "economic_indicators_expert"]
)

financial_analysis_group_summarizer = create_group_summarizer(
    "Phân tích Tài chính (Financial Analysis)",
    ["financial_statement_analyst", "financial_ratio_expert", "valuation_expert", 
     "cash_flow_analyst", "capital_structure_expert"]
)

sectoral_analysis_group_summarizer = create_group_summarizer(
    "Phân tích Ngành (Sectoral Analysis)",
    ["banking_finance_expert", "real_estate_expert", "consumer_goods_expert", 
     "industrial_expert", "technology_expert"]
)

external_factors_group_summarizer = create_group_summarizer(
    "Yếu tố Bên ngoài (External Factors)",
    ["global_markets_expert", "geopolitical_risk_analyst", "regulatory_framework_expert", 
     "monetary_policy_expert", "demographic_trends_expert"]
)

strategy_group_summarizer = create_group_summarizer(
    "Lập chiến lược (Strategy)",
    ["game_theory_strategist", "risk_management_expert", "portfolio_optimization_expert", 
     "asset_allocation_strategist", "investment_psychology_expert"]
)

# Create the final synthesizer node
final_synthesizer = create_final_synthesizer()