import os
import json
import sys
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime

from agent_nodes import AgentState, InputState, OutputState
from group_agents import main_graph

# Thiết lập logging
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

def get_resource_files(resource_dir: str = "../resources") -> List[Path]:
    """
    Get all text files from the resources directory.
    
    Args:
        resource_dir: Path to the resources directory
        
    Returns:
        List of Path objects to text files
    """
    # Get the absolute path to the resources directory
    base_dir = Path(__file__).parent
    full_resource_dir = base_dir / resource_dir
    
    # Check if the directory exists
    if not full_resource_dir.exists():
        logger.warning(f"Resource directory not found: {full_resource_dir}")
        # Create the directory if it doesn't exist
        full_resource_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created empty resource directory: {full_resource_dir}")
        return []
    
    # Get all .txt and .csv files for analysis
    return list(full_resource_dir.glob("*.txt")) + list(full_resource_dir.glob("*.csv"))

def analyze_document(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a single document using the multi-agent system.
    
    Args:
        file_path: Path to the document to analyze
        
    Returns:
        Analysis results for the document
    """
    file_name = file_path.name
    logger.info(f"Starting analysis of document: {file_name}")
    
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize the state for the graph
        initial_state: AgentState = {
            "input_data": content,
            "file_name": file_name,
            "analyses": {},
            "group_summaries": {},
            "final_report": "",
            "search_results": {}
        }
        
        # Run the graph with the initial state
        logger.info(f"Invoking analysis graph for {file_name}")
        start_time = datetime.now()
        result = main_graph.invoke(initial_state)
        end_time = datetime.now()
        
        # Log execution time
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Analysis completed in {duration:.2f} seconds")
        
        # Log search usage statistics
        if "search_results" in result:
            total_searches = 0
            search_by_agent = {}
            
            for key, value in result["search_results"].items():
                if "intermediate_steps" in value:
                    steps = len(value["intermediate_steps"])
                    total_searches += steps
                    search_by_agent[key] = steps
            
            logger.info(f"Total searches performed: {total_searches}")
            logger.info(f"Search distribution: {json.dumps(search_by_agent)}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing document {file_name}: {str(e)}", exc_info=True)
        return {
            "input_data": "",
            "file_name": file_name,
            "analyses": {},
            "group_summaries": {},
            "final_report": f"Error analyzing document: {str(e)}",
            "search_results": {}
        }

def main():
    """Main entry point for the multi-agent investment strategy optimization system."""
    logger.info("=== HỆ THỐNG PHÂN TÍCH VÀ TỐI ƯU HÓA CHIẾN LƯỢC ĐẦU TƯ ===")
    logger.info("Đang tìm kiếm tệp dữ liệu để phân tích...")
    file_paths = get_resource_files()
    
    if not file_paths:
        logger.error("Không tìm thấy tệp dữ liệu nào (*.txt, *.csv) trong thư mục resources.")
        print("Vui lòng thêm các tệp dữ liệu để phân tích và chạy lại.")
        sys.exit(1)
    
    logger.info(f"Đã tìm thấy {len(file_paths)} tệp dữ liệu để phân tích.")
    
    # Create output directory
    output_dir = Path(__file__).parent / "investment_strategies"
    output_dir.mkdir(exist_ok=True)
    
    # Create directory for search results
    search_dir = output_dir / "search_results"
    search_dir.mkdir(exist_ok=True)
    
    # Process each document
    all_results = {}
    for file_path in file_paths:
        file_name = file_path.name
        
        logger.info(f"\nĐang phân tích tệp: {file_name}")
        print("=" * 50)
        result = analyze_document(file_path)
        
        # Save individual result (without search results to keep file size manageable)
        output_file = output_dir / f"{file_name}_strategy.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert state to JSON-serializable format
            serializable_result = {
                "file_name": file_name,
                "analyses": result["analyses"],
                "group_summaries": result["group_summaries"],
                "investment_strategy": result["final_report"]
            }
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        # Save search results separately
        if "search_results" in result:
            search_file = search_dir / f"{file_name}_search_results.json"
            with open(search_file, 'w', encoding='utf-8') as f:
                json.dump(result["search_results"], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Chiến lược đầu tư cho {file_name} đã được lưu tại {output_file}")
        
        # Store result for combined report
        all_results[file_name] = result["final_report"]
        
        # Print final investment strategy for this document
        print("\n=== CHIẾN LƯỢC ĐẦU TƯ ===\n")
        print(result["final_report"])
    
    # Save combined results
    combined_output = output_dir / "all_investment_strategies.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nTất cả chiến lược đầu tư đã được lưu tại {output_dir}")
    logger.info("Phân tích hoàn tất.")

if __name__ == "__main__":
    main()