# enhanced_group_agents.py
import sys
from pathlib import Path
from typing import Dict, List, Any, Annotated, Callable, cast
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.graph import CompiledGraph
from langgraph.pregel import Pregel

sys.path.append(str(Path(__file__).parent))

from agent_nodes import (
    # Expert nodes - Group 1: Market Analysis
    market_analyst_node, technical_analyst_node, fundamental_analyst_node,
    sentiment_analyst_node, economic_indicators_node,
    
    # Expert nodes - Group 2: Financial Analysis
    financial_statement_node, financial_ratio_node, valuation_node,
    cash_flow_node, capital_structure_node,
    
    # Expert nodes - Group 3: Sectoral Analysis
    banking_finance_node, real_estate_node, consumer_goods_node,
    industrial_node, technology_node,
    
    # Expert nodes - Group 4: External Factors
    global_markets_node, geopolitical_risk_node, regulatory_framework_node,
    monetary_policy_node, demographic_trends_node,
    
    # Expert nodes - Group 5: Strategy
    game_theory_node, risk_management_node, portfolio_optimization_node,
    asset_allocation_node, investment_psychology_node,
    
    # Group summarizer nodes
    market_analysis_group_summarizer, financial_analysis_group_summarizer,
    sectoral_analysis_group_summarizer, external_factors_group_summarizer,
    strategy_group_summarizer,
    
    # Final synthesizer
    final_synthesizer,
    
    # State types
    InputState, OutputState, AgentState
)

# Import các tác tử phê bình từ critics.py
from critics import (
    create_group_critic,
    create_meta_critic,
    create_refinement_agent,
    create_group_summary_refinement,
    create_final_report_refinement,
    should_continue_iteration
)

# Định nghĩa cấu trúc nhóm chuyên gia
group_experts = {
    "market_analysis": [
        "market_analyst", "technical_analyst", "fundamental_analyst", 
        "sentiment_analyst", "economic_indicators_expert"
    ],
    "financial_analysis": [
        "financial_statement_analyst", "financial_ratio_expert", "valuation_expert", 
        "cash_flow_analyst", "capital_structure_expert"
    ],
    "sectoral_analysis": [
        "banking_finance_expert", "real_estate_expert", "consumer_goods_expert", 
        "industrial_expert", "technology_expert"
    ],
    "external_factors": [
        "global_markets_expert", "geopolitical_risk_analyst", "regulatory_framework_expert", 
        "monetary_policy_expert", "demographic_trends_expert"
    ],
    "strategy": [
        "game_theory_strategist", "risk_management_expert", "portfolio_optimization_expert", 
        "asset_allocation_strategist", "investment_psychology_expert"
    ]
}

# Tạo các tác tử phê bình cho từng nhóm
market_analysis_critic = create_group_critic("Phân tích Thị trường (Market Analysis)")
financial_analysis_critic = create_group_critic("Phân tích Tài chính (Financial Analysis)")
sectoral_analysis_critic = create_group_critic("Phân tích Ngành (Sectoral Analysis)")
external_factors_critic = create_group_critic("Yếu tố Bên ngoài (External Factors)")
strategy_critic = create_group_critic("Lập chiến lược (Strategy)")

# Tạo tác tử phê bình tổng hợp
meta_critic = create_meta_critic()

# Hàm chuẩn bị trạng thái cho vòng lặp phản hồi
def prepare_iteration_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuẩn bị trạng thái cho một vòng lặp phản hồi mới.
    
    Args:
        state: Trạng thái hiện tại
        
    Returns:
        Trạng thái đã cập nhật cho vòng lặp mới
    """
    # Khởi tạo iteration counter nếu chưa có
    if "iteration" not in state:
        state["iteration"] = 0
    else:
        state["iteration"] += 1
    
    # Thêm thông tin về cấu trúc nhóm cho các tác tử sử dụng
    state["group_experts"] = group_experts
    
    # Khởi tạo các container cho critiques nếu chưa có
    if "critiques" not in state:
        state["critiques"] = {}
    
    print(f"\n[DEBUG] Starting iteration {state['iteration']}")
    
    return state

def create_enhanced_expert_group_graph(
    group_name: str, 
    expert_nodes: Dict[str, Callable], 
    summarizer_node: Callable,
    critic_node: Callable
) -> CompiledGraph:
    """
    Tạo graph nâng cao cho một nhóm chuyên gia với vòng lặp phản hồi.
    
    Args:
        group_name: Tên của nhóm chuyên gia
        expert_nodes: Dict ánh xạ tên chuyên gia đến node function
        summarizer_node: Node function tổng hợp nhóm
        critic_node: Node function phê bình nhóm
        
    Returns:
        CompiledGraph cho nhóm chuyên gia này
    """
    # Tạo workflow cho nhóm
    workflow = StateGraph(AgentState)
    
    # Node chuẩn bị iteration
    workflow.add_node("prepare_iteration", prepare_iteration_state)
    
    # Thêm tất cả expert nodes
    expert_names = list(expert_nodes.keys())
    for expert_name, expert_node in expert_nodes.items():
        workflow.add_node(expert_name, expert_node)
    
    # Thêm summarizer node
    summarizer_name = f"{group_name}_summarizer"
    workflow.add_node(summarizer_name, summarizer_node)
    
    # Thêm critic node
    critic_name = f"{group_name}_critic"
    workflow.add_node(critic_name, critic_node)
    
    # Thêm condition node để kiểm tra điều kiện dừng
    workflow.add_conditional_edges(
        critic_name,
        should_continue_iteration,
        {
            "continue": "prepare_iteration",
            "stop": END
        }
    )
    
    # Kết nối các node
    if expert_names:
        # Kết nối START đến prepare_iteration
        workflow.add_edge(START, "prepare_iteration")
        
        # Kết nối prepare_iteration đến expert đầu tiên
        workflow.add_edge("prepare_iteration", expert_names[0])
        
        # Kết nối các expert lần lượt
        for i in range(len(expert_names) - 1):
            workflow.add_edge(expert_names[i], expert_names[i + 1])
        
        # Kết nối expert cuối cùng đến summarizer
        workflow.add_edge(expert_names[-1], summarizer_name)
    else:
        # Nếu không có expert, kết nối trực tiếp từ prepare_iteration đến summarizer
        workflow.add_edge("prepare_iteration", summarizer_name)
    
    # Kết nối summarizer đến critic
    workflow.add_edge(summarizer_name, critic_name)
    
    # Biên dịch graph
    return workflow.compile()

# Tạo các enhanced group graphs
# Market Analysis group
market_analysis_expert_nodes = {
    "market_analyst": market_analyst_node,
    "technical_analyst": technical_analyst_node,
    "fundamental_analyst": fundamental_analyst_node,
    "sentiment_analyst": sentiment_analyst_node,
    "economic_indicators_expert": economic_indicators_node
}
market_analysis_group_graph = create_enhanced_expert_group_graph(
    "market_analysis", 
    market_analysis_expert_nodes, 
    market_analysis_group_summarizer,
    market_analysis_critic
)

# Financial Analysis group
financial_analysis_expert_nodes = {
    "financial_statement_analyst": financial_statement_node,
    "financial_ratio_expert": financial_ratio_node,
    "valuation_expert": valuation_node,
    "cash_flow_analyst": cash_flow_node,
    "capital_structure_expert": capital_structure_node
}
financial_analysis_group_graph = create_enhanced_expert_group_graph(
    "financial_analysis",
    financial_analysis_expert_nodes,
    financial_analysis_group_summarizer,
    financial_analysis_critic
)

# Sectoral Analysis group
sectoral_analysis_expert_nodes = {
    "banking_finance_expert": banking_finance_node,
    "real_estate_expert": real_estate_node,
    "consumer_goods_expert": consumer_goods_node,
    "industrial_expert": industrial_node,
    "technology_expert": technology_node
}
sectoral_analysis_group_graph = create_enhanced_expert_group_graph(
    "sectoral_analysis",
    sectoral_analysis_expert_nodes,
    sectoral_analysis_group_summarizer,
    sectoral_analysis_critic
)

# External Factors group
external_factors_expert_nodes = {
    "global_markets_expert": global_markets_node,
    "geopolitical_risk_analyst": geopolitical_risk_node,
    "regulatory_framework_expert": regulatory_framework_node,
    "monetary_policy_expert": monetary_policy_node,
    "demographic_trends_expert": demographic_trends_node
}
external_factors_group_graph = create_enhanced_expert_group_graph(
    "external_factors",
    external_factors_expert_nodes,
    external_factors_group_summarizer,
    external_factors_critic
)

# Strategy group (new)
strategy_expert_nodes = {
    "game_theory_strategist": game_theory_node,
    "risk_management_expert": risk_management_node,
    "portfolio_optimization_expert": portfolio_optimization_node,
    "asset_allocation_strategist": asset_allocation_node,
    "investment_psychology_expert": investment_psychology_node
}
strategy_group_graph = create_enhanced_expert_group_graph(
    "strategy",
    strategy_expert_nodes,
    strategy_group_summarizer,
    strategy_critic
)

def create_enhanced_main_graph():
    """
    Tạo main graph nâng cao với vòng lặp phản hồi giữa các nhóm và meta-critic.
    """
    # Tạo main workflow
    main_workflow = StateGraph(AgentState, input=InputState, output=OutputState)
    
    # Thêm node chuẩn bị iteration
    main_workflow.add_node("prepare_main_iteration", prepare_iteration_state)
    
    # Thêm mỗi group graph như một node
    main_workflow.add_node("market_analysis_group", market_analysis_group_graph)
    main_workflow.add_node("financial_analysis_group", financial_analysis_group_graph)
    main_workflow.add_node("sectoral_analysis_group", sectoral_analysis_group_graph)
    main_workflow.add_node("external_factors_group", external_factors_group_graph)
    main_workflow.add_node("strategy_group", strategy_group_graph)
    
    # Thêm final synthesizer node
    main_workflow.add_node("final_synthesis", final_synthesizer)
    
    # Thêm meta-critic node
    main_workflow.add_node("meta_critic", meta_critic)
    
    # Thêm condition node để kiểm tra điều kiện dừng
    main_workflow.add_conditional_edges(
        "meta_critic",
        should_continue_iteration,
        {
            "continue": "prepare_main_iteration",
            "stop": END
        }
    )
    
    # Kết nối các node
    main_workflow.add_edge(START, "prepare_main_iteration")
    main_workflow.add_edge("prepare_main_iteration", "market_analysis_group")
    main_workflow.add_edge("market_analysis_group", "financial_analysis_group")
    main_workflow.add_edge("financial_analysis_group", "sectoral_analysis_group")
    main_workflow.add_edge("sectoral_analysis_group", "external_factors_group")
    main_workflow.add_edge("external_factors_group", "strategy_group")
    main_workflow.add_edge("strategy_group", "final_synthesis")
    main_workflow.add_edge("final_synthesis", "meta_critic")
    
    # Biên dịch main graph
    return main_workflow.compile()

# Tạo main graph
main_graph = create_enhanced_main_graph()

# Hàm tiện ích để chạy phân tích với số lần lặp tối đa
def run_analysis_with_iterations(initial_state: Dict[str, Any], max_iterations: int = 3) -> Dict[str, Any]:
    """
    Chạy phân tích với số lần lặp tối đa
    
    Args:
        initial_state: Trạng thái khởi tạo
        max_iterations: Số lần lặp tối đa
        
    Returns:
        Kết quả phân tích
    """
    # Thêm max_iterations vào state
    initial_state["max_iterations"] = max_iterations
    
    # Chạy graph
    result = main_graph.invoke(initial_state)
    
    return result