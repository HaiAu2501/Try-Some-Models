import os
import sys
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime

from agent_nodes import AgentState
from group_agents import main_graph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("investment_analysis")

# Load environment variables
load_dotenv()

def main():
    """Main entry point for the investment strategy optimization system."""
    logger.info("=== INVESTMENT STRATEGY ANALYSIS AND OPTIMIZATION SYSTEM ===")
    
    # Get user question via console
    print("\nEnter your investment question or goal:")
    user_question = input("> ")
    
    if not user_question.strip():
        logger.error("No question or goal provided.")
        print("Please enter a question or investment goal and run again.")
        sys.exit(1)
    
    logger.info(f"Received question: {user_question}")
    
    # Create output directories
    output_dir = Path(__file__).parent / "investment_strategies"
    output_dir.mkdir(exist_ok=True)
    
    group_output_dir = output_dir / "group_responses"
    group_output_dir.mkdir(exist_ok=True)
    
    expert_output_dir = output_dir / "expert_responses"
    expert_output_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nAnalyzing question: {user_question}")
    print("=" * 50)
    
    # Initialize state
    initial_state = {
        "input_data": user_question,
        "question": user_question,
        "analyses": {},
        "group_1": {},
        "group_2": {},
        "group_3": {},
        "group_4": {},
        "group_5": {},
        "group_summaries": {},
        "final_report": "",
        "search_results": {}
    }
    
    # Run analysis with LangGraph
    logger.info("Calling analysis graph for question")
    start_time = datetime.now()
    result = main_graph.invoke(initial_state)
    end_time = datetime.now()
    
    # Log execution time
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Analysis completed in {duration:.2f} seconds")
    
    # Print final investment strategy to console
    print("\n=== INVESTMENT STRATEGY ===\n")
    print(result["final_report"])
    
    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()