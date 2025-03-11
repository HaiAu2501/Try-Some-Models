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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Now import with relative paths
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

# Load environment variables
load_dotenv()

# Get search tools
search_tools = get_search_tools()

# Define state structures
class InputState(TypedDict):
    input_data: str                  # Data to be analyzed as string
    file_name: str                   # Name of the file being analyzed

class OutputState(TypedDict):
    analyses: Annotated[Dict[str, str], "merge"]  # Analyses from different agents - merge each agent's analysis
    group_summaries: Annotated[Dict[str, str], "merge"]  # Summaries from each group - merge each group's summary
    final_report: str                # Final combined report
    search_results: Annotated[Dict[str, Dict[str, Any]], "merge"]  # Search results from different queries - merge results

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
        return ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("No API key found for OpenAI")

# Extract relevant keywords based on expert type and input data
def extract_relevant_terms(text: str, agent_type: str) -> List[str]:
    """
    Extract relevant terms from input data based on agent type to guide search.
    
    Args:
        text: Input text data
        agent_type: Type of agent/expert
        
    Returns:
        List of relevant terms to focus search
    """
    terms = []
    
    # Extract potential tickers (usually 3-4 uppercase letters)
    import re
    potential_tickers = re.findall(r'\b[A-Z]{3,4}\b', text)
    terms.extend(potential_tickers[:3])  # Limit to first 3 tickers
    
    # Extract sector/industry terms based on agent type
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

# Generic expert agent creation function with tools integration
def create_expert_agent(system_prompt: str, agent_name: str):
    """Create an expert agent with the given system prompt and name, with tools integration."""
    # Define a specific prompt template for agent with tools
    llm = get_model()
    
    # Custom prompt template integrating system prompt with React agent format
    agent_prompt = PromptTemplate.from_template(
        """
{system_prompt}

You have access to the following tools to help with your analysis:

{tools}

Use these tools to research current information about the stock market, companies, 
sectors, and economic indicators to provide an up-to-date analysis.

Follow this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""
    )
    
    # Create the React agent with tools
    agent = create_react_agent(llm, search_tools, agent_prompt)
    
    # Create an agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5
    )
    
    # Wrap the agent execution in the expert analysis function
    def expert_analysis(state: AgentState) -> AgentState:
        """Run expert analysis with tools on input data and store in state."""
        try:
            # Extract values from state
            input_data = state.get("input_data", "")
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running {agent_name} on file: {file_name}")
            
            # Extract relevant terms for search guidance
            relevant_terms = extract_relevant_terms(input_data, agent_name)
            relevant_terms_text = ", ".join(relevant_terms)
            
            # Create the input for the agent
            agent_input = f"""
            Analyze the following data as a {agent_name} for Vietnam stock market investment strategy:
            
            DATA:
            {input_data}
            
            File: {file_name}
            
            Focus on these key areas: {relevant_terms_text}
            
            Provide a detailed analysis from your expert perspective, with specific investment 
            recommendations based on both the provided data and the latest information you can find.
            Cite your sources for any external information.
            """
            
            # Execute the agent with tools
            agent_result = agent_executor.invoke({
                "system_prompt": system_prompt,
                "input": agent_input,
                "tools": search_tools
            })
            
            # Extract the final analysis from the agent
            analysis = agent_result.get("output", "No analysis provided")
            
            # Store search results in state
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
            
        except Exception as e:
            print(f"[ERROR] Error in {agent_name}: {str(e)}")
            analysis = f"Error analyzing with {agent_name}: {str(e)}"
            search_results_for_state = {
                f"{agent_name}_search": {
                    "error": str(e)
                }
            }
        
        # Create a new state with the analysis and search results
        new_state = cast(AgentState, {})
        new_state["analyses"] = {agent_name: analysis}
        new_state["search_results"] = search_results_for_state
        
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

# Create group summarizer nodes with search results integration
def create_group_summarizer(group_name: str, expert_names: List[str]):
    """Create a summarizer for a group of experts with integration of search results."""
    llm = get_model()
    
    # Create a React agent for the group summarizer
    summarizer_system_prompt = f"""
    You are an expert summarizer for the {group_name} group.
    Your task is to synthesize analyses from multiple experts in this group and create
    a comprehensive summary that highlights key insights, areas of agreement, and important differences.
    
    Focus on providing actionable investment recommendations based on the group's collective expertise.
    Incorporate information from both the expert analyses and the latest market data you can find.
    
    Clearly cite your sources for any external information.
    """
    
    # Create agent prompt template
    summarizer_prompt = PromptTemplate.from_template(
        """
{system_prompt}

You have access to the following tools to help with your analysis:

{tools}

Use these tools to research current information relevant to your summary.

Follow this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""
    )
    
    # Create the React agent with tools
    summarizer_agent = create_react_agent(llm, search_tools, summarizer_prompt)
    
    # Create an agent executor
    summarizer_executor = AgentExecutor(
        agent=summarizer_agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=3
    )
    
    def summarize_group(state: AgentState) -> AgentState:
        """Summarize analyses from experts in this group."""
        try:
            # Extract analyses from this group's experts
            expert_analyses = ""
            for expert in expert_names:
                if expert in state.get("analyses", {}):
                    expert_analyses += f"### Analysis from {expert}:\n{state['analyses'][expert]}\n\n"
            
            # Extract search results from experts in this group
            search_insights = ""
            for expert in expert_names:
                search_key = f"{expert}_search"
                if search_key in state.get("search_results", {}):
                    search_data = state["search_results"][search_key]
                    
                    if "intermediate_steps" in search_data:
                        search_insights += f"\n### Search steps by {expert}:\n"
                        
                        for step in search_data["intermediate_steps"]:
                            if "tool" in step and "input" in step:
                                search_insights += f"- Used {step['tool']} to search for: {step['input']}\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running summarizer for {group_name} on file: {file_name}")
            
            # Create the input for the agent
            summarizer_input = f"""
            Create a comprehensive summary of the following expert analyses from the {group_name} group:
            
            {expert_analyses}
            
            File being analyzed: {file_name}
            
            The experts have already searched for the following information:
            {search_insights}
            
            Provide a thorough synthesis of these analyses, highlighting key insights, areas of agreement, 
            and important differences. Use search tools to verify important claims or find additional
            information where needed.
            
            Your summary should focus on actionable investment recommendations for the Vietnam stock market,
            based on both the expert analyses and the latest market data.
            """
            
            # Execute the agent with tools
            summarizer_result = summarizer_executor.invoke({
                "system_prompt": summarizer_system_prompt,
                "input": summarizer_input,
                "tools": search_tools
            })
            
            # Extract the final summary from the agent
            summary = summarizer_result.get("output", "No summary provided")
            
            # Store search results in state
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
            print(f"[ERROR] Error in {group_name} summarizer: {str(e)}")
            summary = f"Error summarizing {group_name}: {str(e)}"
            search_results_for_state = {
                f"{group_name}_summarizer_search": {
                    "error": str(e)
                }
            }
        
        # Create a new state with the group summary and search results
        new_state = cast(AgentState, {})
        new_state["group_summaries"] = {group_name: summary}
        new_state["search_results"] = search_results_for_state
        
        return new_state
    
    return summarize_group

# Create the final report synthesizer with tools integration
def create_final_synthesizer():
    """Create the final synthesizing node that combines all group summaries with search tools."""
    llm = get_model()
    
    # Define system prompt for the final synthesizer
    synthesizer_system_prompt = """
    You are an expert investment strategist specialized in the Vietnam stock market.
    Your task is to synthesize analyses from multiple expert groups and create a comprehensive
    investment strategy report.
    
    Your report should provide clear, actionable investment recommendations including:
    1. Strategic asset allocation
    2. Sector and stock recommendations
    3. Market timing advice
    4. Risk management strategies
    
    Use the tools available to verify important information and ensure your recommendations
    are based on the latest market data and economic indicators.
    
    Provide a well-structured report with specific, actionable insights that investors
    can immediately implement.
    """
    
    # Create agent prompt template
    synthesizer_prompt = PromptTemplate.from_template(
        """
{system_prompt}

You have access to the following tools to help with your analysis:

{tools}

Use these tools to research current information relevant to your investment strategy report.

Follow this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""
    )
    
    # Create the React agent with tools
    synthesizer_agent = create_react_agent(llm, search_tools, synthesizer_prompt)
    
    # Create an agent executor
    synthesizer_executor = AgentExecutor(
        agent=synthesizer_agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5
    )
    
    def synthesize_final_report(state: AgentState) -> AgentState:
        """Create the final synthesized report from all group summaries."""
        try:
            # Format all group summaries
            group_summaries_text = ""
            for group_name, summary in state.get("group_summaries", {}).items():
                group_summaries_text += f"### Summary from {group_name}:\n{summary}\n\n"
            
            # Summarize search usage across all experts and summarizers
            search_usage = ""
            if "search_results" in state:
                # Count the total number of searches
                total_searches = 0
                for search_key, search_data in state.get("search_results", {}).items():
                    if "intermediate_steps" in search_data:
                        total_searches += len(search_data["intermediate_steps"])
                
                search_usage = f"Note: The analysis is based on {total_searches} searches for the latest market information."
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running final synthesizer on file: {file_name}")
            
            # Create the input for the agent
            synthesizer_input = f"""
            Create a comprehensive investment strategy report for the Vietnam stock market based on 
            the following group summaries:
            
            {group_summaries_text}
            
            File being analyzed: {file_name}
            
            {search_usage}
            
            Your report should include:
            
            1. Executive Summary - Key findings and recommendations
            2. Market Analysis - Current state and trends
            3. Investment Strategy:
               a. Strategic Asset Allocation
               b. Recommended Sectors and Stocks
               c. Market Timing Recommendations
               d. Position Sizing and Portfolio Construction
            4. Risk Management Plan
            5. Specific Action Items for Investors
            
            Use search tools to verify important information and ensure your recommendations
            are based on the latest market data. Cite your sources for external information.
            """
            
            # Execute the agent with tools
            synthesizer_result = synthesizer_executor.invoke({
                "system_prompt": synthesizer_system_prompt,
                "input": synthesizer_input,
                "tools": search_tools
            })
            
            # Extract the final report from the agent
            final_report = synthesizer_result.get("output", "No report provided")
            
            # Store search results in state
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
            print(f"[ERROR] Error in final synthesizer: {str(e)}")
            final_report = f"Error generating final report: {str(e)}"
            search_results_for_state = {
                "final_synthesizer_search": {
                    "error": str(e)
                }
            }
        
        # Create a new state with the final report and search results
        new_state = cast(AgentState, {})
        new_state["final_report"] = final_report
        new_state["search_results"] = search_results_for_state
        
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