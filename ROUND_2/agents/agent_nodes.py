from typing import Dict, List, Any, Annotated, TypedDict, cast
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import ToolException
from langchain.prompts import PromptTemplate

# Thêm thư mục cha vào đường dẫn
sys.path.append(str(Path(__file__).parent.parent))

# Bây giờ nhập với đường dẫn tương đối
from search_tools import get_search_tools
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

# Tải biến môi trường
load_dotenv()

# Lấy công cụ tìm kiếm
search_tools = get_search_tools()

# Định nghĩa cấu trúc trạng thái
class InputState(TypedDict):
    input_data: str                  # Dữ liệu cần phân tích dưới dạng chuỗi
    question: str                    # Câu hỏi/mục tiêu người dùng

class OutputState(TypedDict):
    analyses: Annotated[Dict[str, str], "merge"]  # Phân tích từ các tác tử khác nhau (giữ cho tương thích)
    group_1: Annotated[Dict[str, str], "merge"]   # Phân tích từ nhóm 1 - Phân tích Thị trường
    group_2: Annotated[Dict[str, str], "merge"]   # Phân tích từ nhóm 2 - Phân tích Tài chính
    group_3: Annotated[Dict[str, str], "merge"]   # Phân tích từ nhóm 3 - Phân tích Ngành
    group_4: Annotated[Dict[str, str], "merge"]   # Phân tích từ nhóm 4 - Yếu tố Bên ngoài
    group_5: Annotated[Dict[str, str], "merge"]   # Phân tích từ nhóm 5 - Chiến lược
    group_summaries: Annotated[Dict[str, str], "merge"]  # Tổng hợp từ mỗi nhóm
    final_report: str                # Báo cáo tổng hợp cuối cùng
    search_results: Annotated[Dict[str, Dict[str, Any]], "merge"]  # Kết quả tìm kiếm

class AgentState(InputState, OutputState):
    """Trạng thái kết hợp cho hệ thống tác tử, kế thừa cả trạng thái đầu vào và đầu ra."""
    pass

# Hàm để tạo mô hình dựa trên cấu hình môi trường
def get_model():
    """
    Lấy mô hình LLM phù hợp dựa trên cấu hình môi trường.
    Trả về mô hình OpenAI theo mặc định.
    """
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("Không tìm thấy API key cho OpenAI")

# Trích xuất các từ khóa liên quan dựa trên loại chuyên gia và dữ liệu đầu vào
def extract_relevant_terms(text: str, agent_type: str) -> List[str]:
    """
    Trích xuất các thuật ngữ liên quan từ dữ liệu đầu vào dựa trên loại tác tử để hướng dẫn tìm kiếm.
    
    Tham số:
        text: Văn bản đầu vào
        agent_type: Loại tác tử/chuyên gia
        
    Trả về:
        Danh sách các thuật ngữ liên quan để tập trung tìm kiếm
    """
    terms = []
    
    # Trích xuất các mã cổ phiếu tiềm năng (thường là 3-4 chữ cái viết hoa)
    import re
    potential_tickers = re.findall(r'\b[A-Z]{3,4}\b', text)
    terms.extend(potential_tickers[:3])  # Giới hạn ở 3 mã cổ phiếu đầu tiên
    
    # Trích xuất các thuật ngữ ngành/lĩnh vực dựa trên loại tác tử
    if "market" in agent_type:
        sectors = ["market index", "VN-Index", "HNX-Index", "UPCOM", "market trend"]
    elif "technical" in agent_type:
        sectors = ["technical analysis", "chart pattern", "support resistance", "trading volume"]
    elif "fundamental" in agent_type:
        sectors = ["fundamental analysis", "earnings", "valuation", "PE ratio"]
    elif "banking" in agent_type or "financial" in agent_type:
        sectors = ["banking sector", "financial sector", "bank stocks", "interest rates"]
    elif "real_estate" in agent_type:
        sectors = ["real estate market", "property sector", "construction", "housing"]
    elif "consumer" in agent_type:
        sectors = ["consumer goods", "retail", "FMCG", "consumption"]
    elif "technology" in agent_type:
        sectors = ["technology sector", "IT companies", "software", "digital transformation"]
    elif "industrial" in agent_type:
        sectors = ["industrial sector", "manufacturing", "production", "factories"]
    else:
        sectors = ["Vietnam stock market", "investment strategy", "portfolio management"]
    
    terms.extend(sectors)
    return terms

# Hàm tạo tác tử chuyên gia tổng quát với tích hợp công cụ
def create_expert_agent(system_prompt: str, agent_name: str):
    """Tạo một tác tử chuyên gia với prompt hệ thống và tên đã cho, với tích hợp công cụ."""
    # Định nghĩa một mẫu prompt cụ thể cho tác tử với công cụ
    llm = get_model()
    
    # Mẫu prompt tùy chỉnh tích hợp prompt hệ thống với định dạng tác tử React
    agent_prompt = PromptTemplate.from_template(
        """
{system_prompt}

Bạn có quyền truy cập vào các công cụ sau để hỗ trợ phân tích của mình:

{tools}

Sử dụng các công cụ này để nghiên cứu thông tin hiện tại về thị trường chứng khoán, 
các công ty, ngành nghề và chỉ số kinh tế để cung cấp phân tích cập nhật.

Tuân theo định dạng này:

Question: câu hỏi đầu vào bạn phải trả lời
Thought: bạn nên luôn suy nghĩ về việc phải làm gì
Action: hành động cần thực hiện, nên là một trong [{tool_names}]
Action Input: đầu vào cho hành động
Observation: kết quả của hành động
... (quy trình Thought/Action/Action Input/Observation có thể lặp lại nhiều lần)
Thought: Bây giờ tôi đã biết câu trả lời cuối cùng
Final Answer: câu trả lời cuối cùng cho câu hỏi đầu vào ban đầu

Bắt đầu!

Question: {input}
{agent_scratchpad}
"""
    )
    
    # Tạo tác tử React với công cụ
    agent = create_react_agent(llm, search_tools, agent_prompt)
    
    # Tạo trình thực thi tác tử
    agent_executor = AgentExecutor(
        agent=agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5
    )
    
    # Bọc việc thực thi tác tử trong hàm phân tích chuyên gia
    def expert_analysis(state: AgentState) -> AgentState:
        """Chạy phân tích chuyên gia với công cụ trên dữ liệu đầu vào và lưu trữ trong trạng thái."""
        try:
            # Trích xuất giá trị từ trạng thái
            input_data = state.get("input_data", "")
            question = state.get("question", "")
            
            print(f"\n[DEBUG] Đang chạy {agent_name} cho câu hỏi: {question}")
            
            # Trích xuất các thuật ngữ liên quan để hướng dẫn tìm kiếm
            relevant_terms = extract_relevant_terms(input_data, agent_name)
            relevant_terms_text = ", ".join(relevant_terms)
            
            # Tạo đầu vào cho tác tử
            agent_input = f"""
            Với tư cách là {agent_name}, hãy phân tích và trả lời câu hỏi sau về thị trường chứng khoán Việt Nam:
            
            CÂU HỎI NGƯỜI DÙNG:
            {question}
            
            Tập trung vào các lĩnh vực chính này: {relevant_terms_text}
            
            Hãy cung cấp phân tích chi tiết từ góc nhìn chuyên gia của bạn và đưa ra các khuyến nghị đầu tư cụ thể 
            dựa trên câu hỏi của người dùng và thông tin mới nhất bạn có thể tìm thấy.
            Trích dẫn nguồn cho mọi thông tin bên ngoài.
            """
            
            # Thực thi tác tử với công cụ
            agent_result = agent_executor.invoke({
                "system_prompt": system_prompt,
                "input": agent_input,
                "tools": search_tools
            })
            
            # Trích xuất phân tích cuối cùng từ tác tử
            analysis = agent_result.get("output", "Không có phân tích được cung cấp")
            
            # Lưu phân tích ra file txt
            output_dir = Path(__file__).parent / "investment_strategies" / "expert_responses"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            with open(output_dir / f"{agent_name}.txt", 'w', encoding='utf-8') as f:
                f.write(f"=== PHÂN TÍCH TỪ CHUYÊN GIA {agent_name.upper()} ===\n\n")
                f.write(f"Câu hỏi: {question}\n\n")
                f.write(analysis)
            
            # Lưu trữ kết quả tìm kiếm trong trạng thái
            intermediate_steps = agent_result.get("intermediate_steps", [])
            search_results_for_state = {
                f"{agent_name}_search": {
                    "intermediate_steps": [
                        {
                            "tool": step[0].tool,
                            "input": step[0].tool_input,
                            "output": step[1]
                        } for step in intermediate_steps if hasattr(step[0], 'tool')
                    ]
                }
            }
            
            # Xác định nhóm của chuyên gia
            group_key = None
            if agent_name in ["market_analyst", "technical_analyst", "fundamental_analyst", "sentiment_analyst", "economic_indicators_expert"]:
                group_key = "group_1"
            elif agent_name in ["financial_statement_analyst", "financial_ratio_expert", "valuation_expert", "cash_flow_analyst", "capital_structure_expert"]:
                group_key = "group_2"
            elif agent_name in ["banking_finance_expert", "real_estate_expert", "consumer_goods_expert", "industrial_expert", "technology_expert"]:
                group_key = "group_3"
            elif agent_name in ["global_markets_expert", "geopolitical_risk_analyst", "regulatory_framework_expert", "monetary_policy_expert", "demographic_trends_expert"]:
                group_key = "group_4"
            elif agent_name in ["game_theory_strategist", "risk_management_expert", "portfolio_optimization_expert", "asset_allocation_strategist", "investment_psychology_expert"]:
                group_key = "group_5"
            
        except Exception as e:
            print(f"[LỖI] Lỗi trong {agent_name}: {str(e)}")
            analysis = f"Lỗi phân tích với {agent_name}: {str(e)}"
            search_results_for_state = {
                f"{agent_name}_search": {
                    "error": str(e)
                }
            }
            group_key = None
        
        # Tạo trạng thái mới với phân tích và kết quả tìm kiếm
        new_state = cast(AgentState, {})
        new_state["analyses"] = {agent_name: analysis}
        new_state["search_results"] = search_results_for_state
        
        # Thêm phân tích vào nhóm tương ứng nếu có
        if group_key:
            new_state[group_key] = {agent_name: analysis}
        
        return new_state
    
    return expert_analysis

# Tạo các node tác tử chuyên gia cho mỗi nhóm

# Nhóm 1: Phân tích Thị trường
market_analyst_node = create_expert_agent(MARKET_ANALYST, "market_analyst")
technical_analyst_node = create_expert_agent(TECHNICAL_ANALYST, "technical_analyst")
fundamental_analyst_node = create_expert_agent(FUNDAMENTAL_ANALYST, "fundamental_analyst")
sentiment_analyst_node = create_expert_agent(SENTIMENT_ANALYST, "sentiment_analyst")
economic_indicators_node = create_expert_agent(ECONOMIC_INDICATORS_EXPERT, "economic_indicators_expert")

# Nhóm 2: Phân tích Tài chính
financial_statement_node = create_expert_agent(FINANCIAL_STATEMENT_ANALYST, "financial_statement_analyst")
financial_ratio_node = create_expert_agent(FINANCIAL_RATIO_EXPERT, "financial_ratio_expert")
valuation_node = create_expert_agent(VALUATION_EXPERT, "valuation_expert")
cash_flow_node = create_expert_agent(CASH_FLOW_ANALYST, "cash_flow_analyst")
capital_structure_node = create_expert_agent(CAPITAL_STRUCTURE_EXPERT, "capital_structure_expert")

# Nhóm 3: Phân tích Ngành
banking_finance_node = create_expert_agent(BANKING_FINANCE_EXPERT, "banking_finance_expert")
real_estate_node = create_expert_agent(REAL_ESTATE_EXPERT, "real_estate_expert")
consumer_goods_node = create_expert_agent(CONSUMER_GOODS_EXPERT, "consumer_goods_expert")
industrial_node = create_expert_agent(INDUSTRIAL_EXPERT, "industrial_expert")
technology_node = create_expert_agent(TECHNOLOGY_EXPERT, "technology_expert")

# Nhóm 4: Yếu tố Bên ngoài
global_markets_node = create_expert_agent(GLOBAL_MARKETS_EXPERT, "global_markets_expert")
geopolitical_risk_node = create_expert_agent(GEOPOLITICAL_RISK_ANALYST, "geopolitical_risk_analyst")
regulatory_framework_node = create_expert_agent(REGULATORY_FRAMEWORK_EXPERT, "regulatory_framework_expert")
monetary_policy_node = create_expert_agent(MONETARY_POLICY_EXPERT, "monetary_policy_expert")
demographic_trends_node = create_expert_agent(DEMOGRAPHIC_TRENDS_EXPERT, "demographic_trends_expert")

# Nhóm 5: Chiến lược
game_theory_node = create_expert_agent(GAME_THEORY_STRATEGIST, "game_theory_strategist")
risk_management_node = create_expert_agent(RISK_MANAGEMENT_EXPERT, "risk_management_expert")
portfolio_optimization_node = create_expert_agent(PORTFOLIO_OPTIMIZATION_EXPERT, "portfolio_optimization_expert")
asset_allocation_node = create_expert_agent(ASSET_ALLOCATION_STRATEGIST, "asset_allocation_strategist")
investment_psychology_node = create_expert_agent(INVESTMENT_PSYCHOLOGY_EXPERT, "investment_psychology_expert")

# Tạo các node tổng hợp nhóm với tích hợp kết quả tìm kiếm
def create_group_summarizer(group_name: str, expert_names: List[str]):
    """Tạo một trình tổng hợp cho một nhóm chuyên gia với tích hợp kết quả tìm kiếm."""
    llm = get_model()
    
    # Tạo một tác tử React cho trình tổng hợp nhóm
    summarizer_system_prompt = f"""
    Bạn là một chuyên gia tổng hợp cho nhóm {group_name}.
    Nhiệm vụ của bạn là tổng hợp các phân tích từ nhiều chuyên gia trong nhóm này và tạo ra
    một bản tổng hợp toàn diện nêu bật những hiểu biết chính, các lĩnh vực đồng thuận và những khác biệt quan trọng.
    
    Tập trung vào việc trả lời câu hỏi của người dùng một cách trực tiếp và cung cấp các khuyến nghị đầu tư 
    khả thi dựa trên chuyên môn tập thể của nhóm.
    Kết hợp thông tin từ cả phân tích của chuyên gia và dữ liệu thị trường mới nhất bạn có thể tìm thấy.
    
    Trích dẫn rõ ràng nguồn thông tin bên ngoài.
    """
    
    # Tạo mẫu prompt tác tử
    summarizer_prompt = PromptTemplate.from_template(
        """
{system_prompt}

Bạn có quyền truy cập vào các công cụ sau để hỗ trợ phân tích của mình:

{tools}

Sử dụng các công cụ này để nghiên cứu thông tin hiện tại liên quan đến bản tổng hợp của bạn.

Tuân theo định dạng này:

Question: câu hỏi đầu vào bạn phải trả lời
Thought: bạn nên luôn suy nghĩ về việc phải làm gì
Action: hành động cần thực hiện, nên là một trong [{tool_names}]
Action Input: đầu vào cho hành động
Observation: kết quả của hành động
... (quy trình Thought/Action/Action Input/Observation có thể lặp lại nhiều lần)
Thought: Bây giờ tôi đã biết câu trả lời cuối cùng
Final Answer: câu trả lời cuối cùng cho câu hỏi đầu vào ban đầu

Bắt đầu!

Question: {input}
{agent_scratchpad}
"""
    )
    
    # Tạo tác tử React với công cụ
    summarizer_agent = create_react_agent(llm, search_tools, summarizer_prompt)
    
    # Tạo trình thực thi tác tử
    summarizer_executor = AgentExecutor(
        agent=summarizer_agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=3
    )
    
    def summarize_group(state: AgentState) -> AgentState:
        """Tổng hợp phân tích từ các chuyên gia trong nhóm này."""
        try:
            # Trích xuất phân tích từ các chuyên gia của nhóm này
            expert_analyses = ""
            
            # Map group name to group key
            group_mapping = {
                "Phân tích Thị trường (Market Analysis)": "group_1",
                "Phân tích Tài chính (Financial Analysis)": "group_2",
                "Phân tích Ngành (Sectoral Analysis)": "group_3",
                "Yếu tố Bên ngoài (External Factors)": "group_4",
                "Lập chiến lược (Strategy)": "group_5"
            }
            
            group_key = None
            for name, key in group_mapping.items():
                if group_name in name or name in group_name:
                    group_key = key
                    break
            
            # Get analyses from either the group-specific state or the general analyses
            if group_key and group_key in state:
                for expert, analysis in state[group_key].items():
                    expert_analyses += f"### Phân tích từ {expert}:\n{analysis}\n\n"
            else:
                for expert in expert_names:
                    if expert in state.get("analyses", {}):
                        expert_analyses += f"### Phân tích từ {expert}:\n{state['analyses'][expert]}\n\n"
            
            # Trích xuất kết quả tìm kiếm từ các chuyên gia trong nhóm này
            search_insights = ""
            for expert in expert_names:
                search_key = f"{expert}_search"
                if search_key in state.get("search_results", {}):
                    search_data = state["search_results"][search_key]
                    
                    if "intermediate_steps" in search_data:
                        search_insights += f"\n### Các bước tìm kiếm của {expert}:\n"
                        
                        for step in search_data["intermediate_steps"]:
                            if "tool" in step and "input" in step:
                                search_insights += f"- Đã sử dụng {step['tool']} để tìm kiếm: {step['input']}\n"
            
            question = state.get("question", "Không có câu hỏi")
            
            print(f"\n[DEBUG] Đang chạy trình tổng hợp cho {group_name} cho câu hỏi: {question}")
            
            # Tạo đầu vào cho tác tử
            summarizer_input = f"""
            Tạo một bản tổng hợp toàn diện từ các phân tích chuyên gia sau đây từ nhóm {group_name}
            để trả lời câu hỏi của người dùng:
            
            CÂU HỎI NGƯỜI DÙNG:
            {question}
            
            PHÂN TÍCH CỦA CÁC CHUYÊN GIA:
            {expert_analyses}
            
            Các chuyên gia đã tìm kiếm những thông tin sau:
            {search_insights}
            
            Cung cấp một tổng hợp kỹ lưỡng về các phân tích này, nêu bật những hiểu biết chính, 
            các lĩnh vực đồng thuận và những khác biệt quan trọng. Tập trung vào việc trả lời câu hỏi
            của người dùng một cách rõ ràng và cung cấp các khuyến nghị đầu tư khả thi cho thị trường
            chứng khoán Việt Nam dựa trên cả phân tích chuyên gia và dữ liệu thị trường mới nhất.
            """
            
            # Thực thi tác tử với công cụ
            summarizer_result = summarizer_executor.invoke({
                "system_prompt": summarizer_system_prompt,
                "input": summarizer_input,
                "tools": search_tools
            })
            
            # Trích xuất bản tổng hợp cuối cùng từ tác tử
            summary = summarizer_result.get("output", "Không có tổng hợp được cung cấp")
            
            # Lưu bản tổng hợp ra file txt
            output_dir = Path(__file__).parent / "investment_strategies" / "group_responses"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            clean_group_name = group_name.split("(")[0].strip() if "(" in group_name else group_name
            
            with open(output_dir / f"{clean_group_name}.txt", 'w', encoding='utf-8') as f:
                f.write(f"=== TỔNG HỢP TỪ NHÓM {clean_group_name.upper()} ===\n\n")
                f.write(f"Câu hỏi: {question}\n\n")
                f.write(summary)
            
            # Lưu trữ kết quả tìm kiếm trong trạng thái
            intermediate_steps = summarizer_result.get("intermediate_steps", [])
            search_results_for_state = {
                f"{group_name}_summarizer_search": {
                    "intermediate_steps": [
                        {
                            "tool": step[0].tool,
                            "input": step[0].tool_input,
                            "output": step[1]
                        } for step in intermediate_steps if hasattr(step[0], 'tool')
                    ]
                }
            }
            
        except Exception as e:
            print(f"[LỖI] Lỗi trong trình tổng hợp {group_name}: {str(e)}")
            summary = f"Lỗi tổng hợp {group_name}: {str(e)}"
            search_results_for_state = {
                f"{group_name}_summarizer_search": {
                    "error": str(e)
                }
            }
        
        # Tạo trạng thái mới với bản tổng hợp nhóm và kết quả tìm kiếm
        new_state = cast(AgentState, {})
        new_state["group_summaries"] = {group_name: summary}
        new_state["search_results"] = search_results_for_state
        
        return new_state
    
    return summarize_group

# Tạo trình tổng hợp báo cáo cuối cùng với tích hợp công cụ
def create_final_synthesizer():
    """Tạo node tổng hợp cuối cùng kết hợp tất cả các bản tổng hợp nhóm với công cụ tìm kiếm."""
    llm = get_model()
    
    # Định nghĩa prompt hệ thống cho trình tổng hợp cuối cùng
    synthesizer_system_prompt = """
    Bạn là một chuyên gia chiến lược đầu tư chuyên về thị trường chứng khoán Việt Nam.
    Nhiệm vụ của bạn là tổng hợp phân tích từ nhiều nhóm chuyên gia và tạo ra một báo cáo
    chiến lược đầu tư toàn diện để trả lời trực tiếp câu hỏi của người dùng.
    
    Báo cáo của bạn nên cung cấp các khuyến nghị đầu tư rõ ràng, khả thi bao gồm:
    1. Câu trả lời trực tiếp cho câu hỏi của người dùng
    2. Phân bổ tài sản chiến lược
    3. Khuyến nghị về ngành và cổ phiếu
    4. Tư vấn về thời điểm tham gia thị trường
    5. Chiến lược quản lý rủi ro
    
    Sử dụng các công cụ có sẵn để xác minh thông tin quan trọng và đảm bảo khuyến nghị của bạn
    dựa trên dữ liệu thị trường và chỉ số kinh tế mới nhất.
    
    Cung cấp một báo cáo có cấu trúc tốt với những hiểu biết cụ thể, khả thi mà nhà đầu tư
    có thể triển khai ngay lập tức.
    """
    
    # Tạo mẫu prompt tác tử
    synthesizer_prompt = PromptTemplate.from_template(
        """
{system_prompt}

Bạn có quyền truy cập vào các công cụ sau để hỗ trợ phân tích của mình:

{tools}

Sử dụng các công cụ này để nghiên cứu thông tin hiện tại liên quan đến báo cáo chiến lược đầu tư của bạn.

Tuân theo định dạng này:

Question: câu hỏi đầu vào bạn phải trả lời
Thought: bạn nên luôn suy nghĩ về việc phải làm gì
Action: hành động cần thực hiện, nên là một trong [{tool_names}]
Action Input: đầu vào cho hành động
Observation: kết quả của hành động
... (quy trình Thought/Action/Action Input/Observation có thể lặp lại nhiều lần)
Thought: Bây giờ tôi đã biết câu trả lời cuối cùng
Final Answer: câu trả lời cuối cùng cho câu hỏi đầu vào ban đầu

Bắt đầu!

Question: {input}
{agent_scratchpad}
"""
    )
    
    # Tạo tác tử React với công cụ
    synthesizer_agent = create_react_agent(llm, search_tools, synthesizer_prompt)
    
    # Tạo trình thực thi tác tử
    synthesizer_executor = AgentExecutor(
        agent=synthesizer_agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5
    )
    
    def synthesize_final_report(state: AgentState) -> AgentState:
        """Tạo báo cáo tổng hợp cuối cùng từ tất cả các bản tổng hợp nhóm."""
        try:
            # Định dạng tất cả các bản tổng hợp nhóm
            group_summaries_text = ""
            for group_name, summary in state.get("group_summaries", {}).items():
                group_summaries_text += f"### Tổng hợp từ {group_name}:\n{summary}\n\n"
            
            # Tóm tắt việc sử dụng tìm kiếm trên tất cả các chuyên gia và trình tổng hợp
            search_usage = ""
            if "search_results" in state:
                # Đếm tổng số lượng tìm kiếm
                total_searches = 0
                for search_key, search_data in state.get("search_results", {}).items():
                    if "intermediate_steps" in search_data:
                        total_searches += len(search_data["intermediate_steps"])
                
                search_usage = f"Lưu ý: Phân tích này dựa trên {total_searches} lần tìm kiếm thông tin thị trường mới nhất."
            
            question = state.get("question", "Không có câu hỏi")
            
            print(f"\n[DEBUG] Đang chạy trình tổng hợp cuối cùng cho câu hỏi: {question}")
            
            # Tạo đầu vào cho tác tử
            synthesizer_input = f"""
            Tạo một báo cáo chiến lược đầu tư toàn diện cho thị trường chứng khoán Việt Nam
            để trả lời câu hỏi sau của người dùng:
            
            CÂU HỎI NGƯỜI DÙNG:
            {question}
            
            Dựa trên các bản tổng hợp nhóm sau đây:
            
            {group_summaries_text}
            
            {search_usage}
            
            Báo cáo của bạn nên bao gồm:
            
            1. Tóm tắt Điều hành - Câu trả lời trực tiếp cho câu hỏi của người dùng
            2. Phân tích Thị trường - Tình trạng và xu hướng hiện tại
            3. Chiến lược Đầu tư:
               a. Phân bổ Tài sản Chiến lược
               b. Ngành và Cổ phiếu Khuyến nghị
               c. Khuyến nghị về Thời điểm Tham gia Thị trường
               d. Quy mô Vị thế và Xây dựng Danh mục
            4. Kế hoạch Quản lý Rủi ro
            5. Các Hành động Cụ thể cho Nhà đầu tư
            
            Sử dụng công cụ tìm kiếm để xác minh thông tin quan trọng và đảm bảo khuyến nghị của bạn
            dựa trên dữ liệu thị trường mới nhất. Trích dẫn nguồn cho thông tin bên ngoài.
            """
            
            # Thực thi tác tử với công cụ
            synthesizer_result = synthesizer_executor.invoke({
                "system_prompt": synthesizer_system_prompt,
                "input": synthesizer_input,
                "tools": search_tools
            })
            
            # Trích xuất báo cáo cuối cùng từ tác tử
            final_report = synthesizer_result.get("output", "Không có báo cáo được cung cấp")
            
            # Lưu báo cáo ra file txt
            output_dir = Path(__file__).parent / "investment_strategies"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            with open(output_dir / "final_investment_strategy.txt", 'w', encoding='utf-8') as f:
                f.write("=== CHIẾN LƯỢC ĐẦU TƯ TỐI ƯU ===\n\n")
                f.write(f"Câu hỏi: {question}\n\n")
                f.write(final_report)
            
            # Lưu trữ kết quả tìm kiếm trong trạng thái
            intermediate_steps = synthesizer_result.get("intermediate_steps", [])
            search_results_for_state = {
                "final_synthesizer_search": {
                    "intermediate_steps": [
                        {
                            "tool": step[0].tool,
                            "input": step[0].tool_input,
                            "output": step[1]
                        } for step in intermediate_steps if hasattr(step[0], 'tool')
                    ]
                }
            }
            
        except Exception as e:
            print(f"[LỖI] Lỗi trong trình tổng hợp cuối cùng: {str(e)}")
            final_report = f"Lỗi tạo báo cáo cuối cùng: {str(e)}"
            search_results_for_state = {
                "final_synthesizer_search": {
                    "error": str(e)
                }
            }
        
        # Tạo trạng thái mới với báo cáo cuối cùng và kết quả tìm kiếm
        new_state = cast(AgentState, {})
        new_state["final_report"] = final_report
        new_state["search_results"] = search_results_for_state
        
        return new_state
    
    return synthesize_final_report

# Tạo các trình tổng hợp nhóm
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

# Tạo node tổng hợp cuối cùng
final_synthesizer = create_final_synthesizer()