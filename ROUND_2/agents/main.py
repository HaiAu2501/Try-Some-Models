import os
import json
import sys
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime

from agent_nodes import AgentState
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

# Tải biến môi trường
load_dotenv()

def main():
    """Điểm nhập chính cho hệ thống tối ưu hóa chiến lược đầu tư đa tác tử."""
    logger.info("=== HỆ THỐNG PHÂN TÍCH VÀ TỐI ƯU HÓA CHIẾN LƯỢC ĐẦU TƯ ===")
    
    # Nhận câu hỏi từ người dùng qua console
    print("\nNhập câu hỏi hoặc mục tiêu đầu tư của bạn:")
    user_question = input("> ")
    
    if not user_question.strip():
        logger.error("Không có câu hỏi hoặc mục tiêu được cung cấp.")
        print("Vui lòng nhập câu hỏi hoặc mục tiêu đầu tư và chạy lại.")
        sys.exit(1)
    
    logger.info(f"Nhận được câu hỏi: {user_question}")
    
    # Tạo thư mục đầu ra
    output_dir = Path(__file__).parent / "investment_strategies"
    output_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục cho kết quả của từng nhóm
    group_output_dir = output_dir / "group_responses"
    group_output_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục cho kết quả từng chuyên gia
    expert_output_dir = output_dir / "expert_responses"
    expert_output_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục cho kết quả tìm kiếm
    search_dir = output_dir / "search_results"
    search_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nĐang phân tích câu hỏi: {user_question}")
    print("=" * 50)
    
    # Khởi tạo trạng thái cho đồ thị
    initial_state: AgentState = {
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
    
    # Chạy đồ thị với trạng thái ban đầu
    logger.info(f"Đang gọi đồ thị phân tích cho câu hỏi")
    start_time = datetime.now()
    result = main_graph.invoke(initial_state)
    end_time = datetime.now()
    
    # Ghi lại thời gian thực thi
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Phân tích hoàn thành trong {duration:.2f} giây")
    
    # Ghi kết quả từng nhóm ra file txt
    group_mapping = {
        "market_analysis": ("Phân tích Thị trường", "group_1"),
        "financial_analysis": ("Phân tích Tài chính", "group_2"),
        "sectoral_analysis": ("Phân tích Ngành", "group_3"),
        "external_factors": ("Yếu tố Bên ngoài", "group_4"),
        "strategy": ("Lập chiến lược", "group_5")
    }
    
    for group_name, (group_title, group_key) in group_mapping.items():
        if group_name in result.get("group_summaries", {}):
            group_summary = result["group_summaries"][group_name]
            group_file = group_output_dir / f"{group_name}.txt"
            
            with open(group_file, 'w', encoding='utf-8') as f:
                f.write(f"=== PHÂN TÍCH TỪ NHÓM {group_title.upper()} ===\n\n")
                f.write(f"Câu hỏi: {user_question}\n\n")
                f.write(group_summary)
            
            logger.info(f"Đã lưu phân tích của nhóm {group_name} tại {group_file}")
    
    # Lưu báo cáo tổng hợp
    report_file = output_dir / "investment_strategy.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== CHIẾN LƯỢC ĐẦU TƯ TỐI ƯU ===\n\n")
        f.write(f"Câu hỏi: {user_question}\n\n")
        f.write(result["final_report"])
    
    logger.info(f"Chiến lược đầu tư đã được lưu tại {report_file}")
    
    # In chiến lược đầu tư cuối cùng ra console
    print("\n=== CHIẾN LƯỢC ĐẦU TƯ ===\n")
    print(result["final_report"])
    
    logger.info("Phân tích hoàn tất.")

if __name__ == "__main__":
    main()