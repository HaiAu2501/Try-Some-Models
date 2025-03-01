import sys
from pathlib import Path
from typing import Dict, List, Any, Annotated, Callable
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START

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

def create_expert_group_graph(group_name: str, expert_nodes: Dict[str, Callable], summarizer_node: Callable) -> StateGraph:
    """
    Create a graph for a group of expert agents that work sequentially and then are summarized.
    
    Args:
        group_name: Name of the expert group
        expert_nodes: Dictionary mapping expert names to their node functions
        summarizer_node: The node function that summarizes this group's outputs
    
    Returns:
        A StateGraph for this expert group
    """
    # Create a new graph for this group
    workflow = StateGraph(AgentState)
    
    # Add all expert nodes
    expert_names = list(expert_nodes.keys())
    for expert_name, expert_node in expert_nodes.items():
        workflow.add_node(expert_name, expert_node)
    
    # Add the summarizer node
    summarizer_name = f"{group_name}_summarizer"
    workflow.add_node(summarizer_name, summarizer_node)
    
    # Connect experts sequentially
    if expert_names:
        # Connect START to the first expert
        workflow.add_edge(START, expert_names[0])
        
        # Connect each expert to the next one
        for i in range(len(expert_names) - 1):
            workflow.add_edge(expert_names[i], expert_names[i + 1])
        
        # Connect the last expert to the summarizer
        workflow.add_edge(expert_names[-1], summarizer_name)
    else:
        # If no experts, connect START directly to summarizer (unlikely case)
        workflow.add_edge(START, summarizer_name)
    
    # Connect the summarizer to the end of this group's workflow
    workflow.add_edge(summarizer_name, END)
    
    # Compile the graph
    return workflow.compile()

# Create the academic quantitative group graph
academic_expert_nodes = {
    "econometrician": econometrician_node,
    "empirical_economist": empirical_economist_node,
    "normative_economist": normative_economist_node,
    "macroeconomist": macroeconomist_node,
    "microeconomist": microeconomist_node
}
academic_group_graph = create_expert_group_graph(
    "academic_quantitative", 
    academic_expert_nodes, 
    academic_group_summarizer
)

# Create the behavioral social group graph
behavioral_expert_nodes = {
    "behavioral_economist": behavioral_economist_node,
    "socio_economist": socio_economist_node
}
behavioral_group_graph = create_expert_group_graph(
    "behavioral_social",
    behavioral_expert_nodes,
    behavioral_group_summarizer
)

# Create the market business group graph
market_expert_nodes = {
    "corporate_management": corporate_management_node,
    "financial_economist": financial_economist_node,
    "international_economist": international_economist_node,
    "logistics_expert": logistics_node,
    "trade_commerce_expert": trade_commerce_node
}
market_group_graph = create_expert_group_graph(
    "market_business",
    market_expert_nodes,
    market_group_summarizer
)

# Create the policy innovation group graph
policy_expert_nodes = {
    "digital_economy_expert": digital_economy_node,
    "environmental_economist": environmental_economist_node,
    "public_policy_expert": public_policy_node
}
policy_group_graph = create_expert_group_graph(
    "policy_innovation",
    policy_expert_nodes,
    policy_group_summarizer
)

def create_main_graph():
    """
    Create the main graph that coordinates all group graphs and produces the final report.
    """
    # Create the main workflow
    main_workflow = StateGraph(AgentState, input=InputState, output=OutputState)
    
    # Add each group graph as a node
    main_workflow.add_node("academic_group", academic_group_graph)
    main_workflow.add_node("behavioral_group", behavioral_group_graph)
    main_workflow.add_node("market_group", market_group_graph)
    main_workflow.add_node("policy_group", policy_group_graph)
    
    # Add the final synthesizer node
    main_workflow.add_node("final_synthesis", final_synthesizer)
    
    # Run groups sequentially instead of in parallel
    main_workflow.add_edge(START, "academic_group")
    main_workflow.add_edge("academic_group", "behavioral_group")
    main_workflow.add_edge("behavioral_group", "market_group")
    main_workflow.add_edge("market_group", "policy_group")
    main_workflow.add_edge("policy_group", "final_synthesis")
    
    # Connect the final synthesizer to the end
    main_workflow.add_edge("final_synthesis", END)
    
    # Compile the main graph
    return main_workflow.compile()

# Create the main graph
main_graph = create_main_graph()