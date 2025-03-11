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
    # Expert nodes
    econometrician_node, empirical_economist_node, normative_economist_node,
    macroeconomist_node, microeconomist_node,
    behavioral_economist_node, socio_economist_node,
    corporate_management_node, financial_economist_node, international_economist_node,
    logistics_node, trade_commerce_node,
    digital_economy_node, environmental_economist_node, public_policy_node,
    
    # Group summarizer nodes
    academic_group_summarizer, behavioral_group_summarizer,
    market_group_summarizer, policy_group_summarizer,
    
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
    "academic_quantitative": [
        "econometrician", "empirical_economist", "normative_economist", 
        "macroeconomist", "microeconomist"
    ],
    "behavioral_social": [
        "behavioral_economist", "socio_economist"
    ],
    "market_business": [
        "corporate_management", "financial_economist", "international_economist", 
        "logistics_expert", "trade_commerce_expert"
    ],
    "policy_innovation": [
        "digital_economy_expert", "environmental_economist", "public_policy_expert"
    ]
}

# Tạo các tác tử phê bình cho từng nhóm
academic_critic = create_group_critic("Học Thuật Định Lượng (Academic Quantitative)")
behavioral_critic = create_group_critic("Hành Vi Xã Hội (Behavioral Social)")
market_critic = create_group_critic("Thị Trường Doanh Nghiệp (Market Business)")
policy_critic = create_group_critic("Chính Sách Đổi Mới (Policy Innovation)")

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
# Academic Quantitative group
academic_expert_nodes = {
    "econometrician": econometrician_node,
    "empirical_economist": empirical_economist_node,
    "normative_economist": normative_economist_node,
    "macroeconomist": macroeconomist_node,
    "microeconomist": microeconomist_node
}
academic_group_graph = create_enhanced_expert_group_graph(
    "academic_quantitative", 
    academic_expert_nodes, 
    academic_group_summarizer,
    academic_critic
)

# Behavioral Social group
behavioral_expert_nodes = {
    "behavioral_economist": behavioral_economist_node,
    "socio_economist": socio_economist_node
}
behavioral_group_graph = create_enhanced_expert_group_graph(
    "behavioral_social",
    behavioral_expert_nodes,
    behavioral_group_summarizer,
    behavioral_critic
)

# Market Business group
market_expert_nodes = {
    "corporate_management": corporate_management_node,
    "financial_economist": financial_economist_node,
    "international_economist": international_economist_node,
    "logistics_expert": logistics_node,
    "trade_commerce_expert": trade_commerce_node
}
market_group_graph = create_enhanced_expert_group_graph(
    "market_business",
    market_expert_nodes,
    market_group_summarizer,
    market_critic
)

# Policy Innovation group
policy_expert_nodes = {
    "digital_economy_expert": digital_economy_node,
    "environmental_economist": environmental_economist_node,
    "public_policy_expert": public_policy_node
}
policy_group_graph = create_enhanced_expert_group_graph(
    "policy_innovation",
    policy_expert_nodes,
    policy_group_summarizer,
    policy_critic
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
    main_workflow.add_node("academic_group", academic_group_graph)
    main_workflow.add_node("behavioral_group", behavioral_group_graph)
    main_workflow.add_node("market_group", market_group_graph)
    main_workflow.add_node("policy_group", policy_group_graph)
    
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
    main_workflow.add_edge("prepare_main_iteration", "academic_group")
    main_workflow.add_edge("academic_group", "behavioral_group")
    main_workflow.add_edge("behavioral_group", "market_group")
    main_workflow.add_edge("market_group", "policy_group")
    main_workflow.add_edge("policy_group", "final_synthesis")
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