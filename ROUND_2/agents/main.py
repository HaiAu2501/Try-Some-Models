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

# Tải biến môi trường
load_dotenv()

def get_resource_files(resource_dir: str = "../resources") -> List[Path]:
    """
    Lấy tất cả các tệp văn bản từ thư mục tài nguyên.
    
    Tham số:
        resource_dir: Đường dẫn đến thư mục tài nguyên
        
    Trả về:
        Danh sách các đối tượng Path đến các tệp văn bản
    """
    # Lấy đường dẫn tuyệt đối đến thư mục tài nguyên
    base_dir = Path(__file__).parent
    full_resource_dir = base_dir / resource_dir
    
    # Kiểm tra xem thư mục có tồn tại không
    if not full_resource_dir.exists():
        logger.warning(f"Không tìm thấy thư mục tài nguyên: {full_resource_dir}")
        # Tạo thư mục nếu không tồn tại
        full_resource_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Đã tạo thư mục tài nguyên trống: {full_resource_dir}")
        return []
    
    # Lấy tất cả các tệp .txt và .csv để phân tích
    return list(full_resource_dir.glob("*.txt")) + list(full_resource_dir.glob("*.csv"))

def analyze_document(file_path: Path) -> Dict[str, Any]:
    """
    Phân tích một tài liệu duy nhất bằng hệ thống đa tác tử.
    
    Tham số:
        file_path: Đường dẫn đến tài liệu cần phân tích
        
    Trả về:
        Kết quả phân tích cho tài liệu
    """
    file_name = file_path.name
    logger.info(f"Bắt đầu phân tích tài liệu: {file_name}")
    
    try:
        # Đọc nội dung tệp
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Khởi tạo trạng thái cho đồ thị
        initial_state: AgentState = {
            "input_data": content,
            "file_name": file_name,
            "analyses": {},
            "group_summaries": {},
            "final_report": "",
            "search_results": {}
        }
        
        # Chạy đồ thị với trạng thái ban đầu
        logger.info(f"Đang gọi đồ thị phân tích cho {file_name}")
        start_time = datetime.now()
        result = main_graph.invoke(initial_state)
        end_time = datetime.now()
        
        # Ghi lại thời gian thực thi
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Phân tích hoàn thành trong {duration:.2f} giây")
        
        # Ghi lại thống kê sử dụng tìm kiếm
        if "search_results" in result:
            total_searches = 0
            search_by_agent = {}
            
            for key, value in result["search_results"].items():
                if "intermediate_steps" in value:
                    steps = len(value["intermediate_steps"])
                    total_searches += steps
                    search_by_agent[key] = steps
            
            logger.info(f"Tổng số lần tìm kiếm đã thực hiện: {total_searches}")
            logger.info(f"Phân phối tìm kiếm: {json.dumps(search_by_agent)}")
        
        return result
    
    except Exception as e:
        logger.error(f"Lỗi khi phân tích tài liệu {file_name}: {str(e)}", exc_info=True)
        return {
            "input_data": "",
            "file_name": file_name,
            "analyses": {},
            "group_summaries": {},
            "final_report": f"Lỗi phân tích tài liệu: {str(e)}",
            "search_results": {}
        }

def main():
    """Điểm nhập chính cho hệ thống tối ưu hóa chiến lược đầu tư đa tác tử."""
    logger.info("=== HỆ THỐNG PHÂN TÍCH VÀ TỐI ƯU HÓA CHIẾN LƯỢC ĐẦU TƯ ===")
    logger.info("Đang tìm kiếm tệp dữ liệu để phân tích...")
    file_paths = get_resource_files()
    
    if not file_paths:
        logger.error("Không tìm thấy tệp dữ liệu nào (*.txt, *.csv) trong thư mục resources.")
        print("Vui lòng thêm các tệp dữ liệu để phân tích và chạy lại.")
        sys.exit(1)
    
    logger.info(f"Đã tìm thấy {len(file_paths)} tệp dữ liệu để phân tích.")
    
    # Tạo thư mục đầu ra
    output_dir = Path(__file__).parent / "investment_strategies"
    output_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục cho kết quả tìm kiếm
    search_dir = output_dir / "search_results"
    search_dir.mkdir(exist_ok=True)
    
    # Xử lý từng tài liệu
    all_results = {}
    for file_path in file_paths:
        file_name = file_path.name
        
        logger.info(f"\nĐang phân tích tệp: {file_name}")
        print("=" * 50)
        result = analyze_document(file_path)
        
        # Lưu kết quả riêng lẻ (không có kết quả tìm kiếm để giữ kích thước tệp ở mức hợp lý)
        output_file = output_dir / f"{file_name}_strategy.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Chuyển đổi trạng thái thành định dạng có thể serialize thành JSON
            serializable_result = {
                "file_name": file_name,
                "analyses": result["analyses"],
                "group_summaries": result["group_summaries"],
                "investment_strategy": result["final_report"]
            }
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        # Lưu kết quả tìm kiếm riêng biệt
        if "search_results" in result:
            search_file = search_dir / f"{file_name}_search_results.json"
            with open(search_file, 'w', encoding='utf-8') as f:
                json.dump(result["search_results"], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Chiến lược đầu tư cho {file_name} đã được lưu tại {output_file}")
        
        # Lưu trữ kết quả cho báo cáo tổng hợp
        all_results[file_name] = result["final_report"]
        
        # In chiến lược đầu tư cuối cùng cho tài liệu này
        print("\n=== CHIẾN LƯỢC ĐẦU TƯ ===\n")
        print(result["final_report"])
    
    # Lưu kết quả tổng hợp
    combined_output = output_dir / "all_investment_strategies.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nTất cả chiến lược đầu tư đã được lưu tại {output_dir}")
    logger.info("Phân tích hoàn tất.")

if __name__ == "__main__":
    main()