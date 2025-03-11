# critics.py
from typing import Dict, List, Any, Annotated, TypedDict, cast
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document
from langchain_openai import ChatOpenAI

# Định nghĩa các prompt cho Critic Agent
GROUP_CRITIC_PROMPT = """
<role>
Bạn là Tác tử Phê bình (Critic Agent) cho nhóm chuyên gia {group_name}.
</role>

<task>
Nhiệm vụ của bạn là:
1. Đánh giá phân tích của các chuyên gia và tổng hợp của nhóm một cách khách quan
2. Chỉ ra các điểm yếu, thiếu sót, mâu thuẫn trong phân tích
3. Nhận diện các khía cạnh quan trọng của dữ liệu đã bị bỏ qua
4. Đề xuất hướng cải thiện cụ thể cho từng chuyên gia

Hãy tập trung vào:
- Tính nhất quán giữa các phân tích của các chuyên gia
- Độ sâu và tính toàn diện của phân tích
- Hiệu quả của việc tích hợp các góc nhìn chuyên môn khác nhau
- Tính thực tiễn của các khuyến nghị

Khi đánh giá, hãy phân loại các vấn đề thành:
- Mức độ nghiêm trọng cao: Cần phải sửa chữa ngay lập tức
- Mức độ trung bình: Đòi hỏi sự cải thiện đáng kể
- Mức độ thấp: Có thể cải thiện thêm nếu có thời gian
</task>
"""

META_CRITIC_PROMPT = """
<role>
Bạn là Tác tử Phê bình Tổng hợp (Meta-Critic Agent).
</role>

<task>
Nhiệm vụ của bạn là:
1. Đánh giá báo cáo tổng hợp cuối cùng từ tất cả các nhóm chuyên gia
2. Xác định mâu thuẫn, chồng chéo hoặc thiếu sót giữa các nhóm
3. Nhận diện các góc nhìn quan trọng đã bị bỏ qua trong phân tích tổng thể
4. Đánh giá tính hiệu quả của việc tích hợp các quan điểm khác nhau
5. Chỉ ra các cải tiến cần thiết cho báo cáo cuối cùng

Hãy tập trung đánh giá:
- Sự cân bằng giữa các góc nhìn khác nhau
- Tính nhất quán của toàn bộ phân tích
- Khả năng áp dụng và tính thực tiễn của các khuyến nghị
- Mức độ bao phủ các vấn đề then chốt từ dữ liệu nguồn

Đưa ra phản hồi cụ thể cho những khía cạnh nào cần được phân tích sâu hơn, và những khuyến nghị nào cần được làm rõ hoặc thêm chi tiết.
</task>
"""

# Hàm để lấy model LLM
def get_model():
    """
    Get the appropriate LLM model based on environment configuration.
    Returns OpenAI model by default.
    """
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("No API key found for OpenAI")

# Tạo tác tử phê bình cho nhóm
def create_group_critic(group_name: str):
    """
    Tạo tác tử phê bình (critic) cho một nhóm chuyên gia.
    
    Args:
        group_name: Tên của nhóm chuyên gia
        
    Returns:
        Hàm critic thực hiện đánh giá phân tích của nhóm
    """
    model = get_model()
    system_prompt = GROUP_CRITIC_PROMPT.format(group_name=group_name)
    
    def critic_evaluation(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Đánh giá phân tích từ các chuyên gia và tổng hợp của nhóm.
        
        Args:
            state: Trạng thái hiện tại của hệ thống
            
        Returns:
            Trạng thái mới với phản hồi của critic
        """
        try:
            # Trích xuất dữ liệu từ state
            expert_analyses = ""
            group_summary = ""
            
            # Tập hợp các phân tích của chuyên gia trong nhóm
            for expert, analysis in state.get("analyses", {}).items():
                if expert in state.get("group_experts", {}).get(group_name, []):
                    expert_analyses += f"### Phân tích từ {expert}:\n{analysis}\n\n"
            
            # Lấy tổng hợp của nhóm
            if group_name in state.get("group_summaries", {}):
                group_summary = state["group_summaries"][group_name]
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running critic for {group_name} on file: {file_name}")
            
            # Tạo messages cho model
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Hãy đánh giá các phân tích sau đây từ các chuyên gia của nhóm {group_name}:
                
                {expert_analyses}
                
                Tổng hợp của nhóm:
                {group_summary}
                
                Tài liệu đang phân tích có tên: {file_name}
                
                Hãy đưa ra phê bình chi tiết về phân tích của nhóm, chỉ ra các điểm yếu và đề xuất cải tiến cụ thể.
                """)
            ]
            
            # Gọi model
            response = model.invoke(messages)
            critique = response.content
            
        except Exception as e:
            print(f"[ERROR] Error in {group_name} critic: {str(e)}")
            critique = f"Error generating critique for {group_name}: {str(e)}"
        
        # Tạo state mới với phê bình của critic
        new_state = cast(Dict[str, Any], {})
        if "critiques" not in state:
            new_state["critiques"] = {}
        
        new_state["critiques"] = {group_name: critique}
        
        return new_state
    
    return critic_evaluation

# Tạo tác tử phê bình tổng hợp (meta-critic)
def create_meta_critic():
    """
    Tạo tác tử phê bình tổng hợp (meta-critic) đánh giá báo cáo cuối cùng.
    
    Returns:
        Hàm meta-critic thực hiện đánh giá tổng thể
    """
    model = get_model()
    
    def meta_critic_evaluation(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Đánh giá báo cáo tổng hợp cuối cùng.
        
        Args:
            state: Trạng thái hiện tại của hệ thống
            
        Returns:
            Trạng thái mới với phản hồi của meta-critic
        """
        try:
            # Trích xuất dữ liệu từ state
            group_summaries_text = ""
            final_report = state.get("final_report", "")
            
            # Tập hợp các tổng hợp từ các nhóm
            for group_name, summary in state.get("group_summaries", {}).items():
                group_summaries_text += f"### Tổng kết từ nhóm {group_name}:\n{summary}\n\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running meta-critic on file: {file_name}")
            
            # Tạo messages cho model
            messages = [
                SystemMessage(content=META_CRITIC_PROMPT),
                HumanMessage(content=f"""
                Hãy đánh giá báo cáo tổng hợp cuối cùng và các tổng kết từ các nhóm:
                
                ## Tổng kết từ các nhóm:
                {group_summaries_text}
                
                ## Báo cáo cuối cùng:
                {final_report}
                
                Tài liệu đang phân tích có tên: {file_name}
                
                Hãy đưa ra đánh giá toàn diện về báo cáo cuối cùng, chỉ ra các mâu thuẫn, thiếu sót và hướng cải thiện.
                """)
            ]
            
            # Gọi model
            response = model.invoke(messages)
            meta_critique = response.content
            
        except Exception as e:
            print(f"[ERROR] Error in meta-critic: {str(e)}")
            meta_critique = f"Error generating meta-critique: {str(e)}"
        
        # Tạo state mới với phê bình của meta-critic
        new_state = cast(Dict[str, Any], {})
        new_state["meta_critique"] = meta_critique
        
        return new_state
    
    return meta_critic_evaluation

# Hàm tinh chỉnh phân tích dựa trên phê bình
def create_refinement_agent(expert_name: str, expert_system_prompt: str):
    """
    Tạo tác tử tinh chỉnh phân tích dựa trên phê bình.
    
    Args:
        expert_name: Tên của chuyên gia
        expert_system_prompt: System prompt gốc của chuyên gia
        
    Returns:
        Hàm refinement agent thực hiện tinh chỉnh phân tích
    """
    model = get_model()
    
    def refine_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tinh chỉnh phân tích dựa trên phê bình.
        
        Args:
            state: Trạng thái hiện tại của hệ thống
            
        Returns:
            Trạng thái mới với phân tích đã tinh chỉnh
        """
        try:
            # Trích xuất dữ liệu từ state
            input_data = state.get("input_data", "")
            file_name = state.get("file_name", "Unknown file")
            
            # Lấy phân tích hiện tại của chuyên gia
            current_analysis = state.get("analyses", {}).get(expert_name, "")
            
            # Lấy phê bình liên quan đến chuyên gia
            critiques = ""
            for group_name, critique in state.get("critiques", {}).items():
                # Kiểm tra xem chuyên gia này có thuộc nhóm đang xét không
                if expert_name in state.get("group_experts", {}).get(group_name, []):
                    critiques += f"### Phê bình từ nhóm {group_name}:\n{critique}\n\n"
            
            # Nếu có meta-critique, thêm vào
            if "meta_critique" in state:
                critiques += f"### Phê bình tổng thể:\n{state['meta_critique']}\n\n"
            
            print(f"\n[DEBUG] Refining analysis for {expert_name} on file: {file_name}")
            
            # Nếu không có phê bình, giữ nguyên phân tích
            if not critiques:
                return {"analyses": {expert_name: current_analysis}}
            
            # Tạo messages cho model
            messages = [
                SystemMessage(content=f"""
                {expert_system_prompt}
                
                Bạn cần tinh chỉnh phân tích trước đó dựa trên phê bình nhận được.
                Hãy giải quyết các vấn đề được nêu ra trong phê bình và cải thiện chất lượng phân tích.
                """),
                HumanMessage(content=f"""
                Dữ liệu cần phân tích:
                
                {input_data}
                
                Tài liệu có tên: {file_name}
                
                Phân tích hiện tại của bạn:
                {current_analysis}
                
                Phê bình nhận được:
                {critiques}
                
                Hãy tinh chỉnh phân tích của bạn dựa trên phê bình trên. Đảm bảo giải quyết các điểm yếu và bổ sung các góc nhìn còn thiếu.
                """)
            ]
            
            # Gọi model
            response = model.invoke(messages)
            refined_analysis = response.content
            
        except Exception as e:
            print(f"[ERROR] Error in refinement for {expert_name}: {str(e)}")
            refined_analysis = current_analysis  # Giữ nguyên phân tích cũ nếu có lỗi
        
        # Tạo state mới với phân tích đã tinh chỉnh
        new_state = cast(Dict[str, Any], {})
        new_state["analyses"] = {expert_name: refined_analysis}
        
        return new_state
    
    return refine_analysis

# Hàm tinh chỉnh tổng hợp nhóm dựa trên phê bình
def create_group_summary_refinement(group_name: str):
    """
    Tạo tác tử tinh chỉnh tổng hợp nhóm dựa trên phê bình.
    
    Args:
        group_name: Tên của nhóm
        
    Returns:
        Hàm refinement agent thực hiện tinh chỉnh tổng hợp nhóm
    """
    model = get_model()
    
    def refine_group_summary(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tinh chỉnh tổng hợp nhóm dựa trên phê bình.
        
        Args:
            state: Trạng thái hiện tại của hệ thống
            
        Returns:
            Trạng thái mới với tổng hợp nhóm đã tinh chỉnh
        """
        try:
            # Trích xuất dữ liệu từ state
            expert_analyses = ""
            group_critique = state.get("critiques", {}).get(group_name, "")
            current_summary = state.get("group_summaries", {}).get(group_name, "")
            
            # Nếu không có phê bình, giữ nguyên tổng hợp
            if not group_critique:
                return {"group_summaries": {group_name: current_summary}}
            
            # Tập hợp các phân tích của chuyên gia trong nhóm
            for expert, analysis in state.get("analyses", {}).items():
                if expert in state.get("group_experts", {}).get(group_name, []):
                    expert_analyses += f"### Phân tích từ {expert}:\n{analysis}\n\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Refining summary for group {group_name} on file: {file_name}")
            
            # Tạo messages cho model
            messages = [
                SystemMessage(content=f"""
                Bạn là người tổng hợp ý kiến cho nhóm chuyên gia {group_name}.
                Nhiệm vụ của bạn là tinh chỉnh tổng hợp trước đó dựa trên phê bình nhận được.
                Hãy giải quyết các vấn đề được nêu ra trong phê bình và cải thiện chất lượng tổng hợp.
                """),
                HumanMessage(content=f"""
                Các phân tích từ các chuyên gia trong nhóm:
                
                {expert_analyses}
                
                Tổng hợp hiện tại của nhóm:
                {current_summary}
                
                Phê bình nhận được:
                {group_critique}
                
                Tài liệu đang phân tích có tên: {file_name}
                
                Hãy tinh chỉnh tổng hợp của nhóm dựa trên phê bình trên. Đảm bảo giải quyết các điểm yếu và bổ sung các góc nhìn còn thiếu.
                """)
            ]
            
            # Gọi model
            response = model.invoke(messages)
            refined_summary = response.content
            
        except Exception as e:
            print(f"[ERROR] Error in summary refinement for {group_name}: {str(e)}")
            refined_summary = current_summary  # Giữ nguyên tổng hợp cũ nếu có lỗi
        
        # Tạo state mới với tổng hợp đã tinh chỉnh
        new_state = cast(Dict[str, Any], {})
        new_state["group_summaries"] = {group_name: refined_summary}
        
        return new_state
    
    return refine_group_summary

# Hàm tinh chỉnh báo cáo cuối cùng dựa trên phê bình tổng thể
def create_final_report_refinement():
    """
    Tạo tác tử tinh chỉnh báo cáo cuối cùng dựa trên phê bình tổng thể.
    
    Returns:
        Hàm refinement agent thực hiện tinh chỉnh báo cáo cuối cùng
    """
    model = get_model()
    
    def refine_final_report(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tinh chỉnh báo cáo cuối cùng dựa trên phê bình tổng thể.
        
        Args:
            state: Trạng thái hiện tại của hệ thống
            
        Returns:
            Trạng thái mới với báo cáo cuối cùng đã tinh chỉnh
        """
        try:
            # Trích xuất dữ liệu từ state
            meta_critique = state.get("meta_critique", "")
            current_report = state.get("final_report", "")
            
            # Nếu không có phê bình tổng thể, giữ nguyên báo cáo
            if not meta_critique:
                return {"final_report": current_report}
            
            # Format tất cả tổng hợp nhóm
            group_summaries_text = ""
            for group_name, summary in state.get("group_summaries", {}).items():
                group_summaries_text += f"### Tổng kết từ nhóm {group_name}:\n{summary}\n\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Refining final report on file: {file_name}")
            
            # Tạo messages cho model
            messages = [
                SystemMessage(content="""
                Bạn là chuyên gia tổng hợp báo cáo cuối cùng.
                Nhiệm vụ của bạn là tinh chỉnh báo cáo trước đó dựa trên phê bình tổng thể nhận được.
                Hãy giải quyết các vấn đề được nêu ra trong phê bình và cải thiện chất lượng báo cáo.
                """),
                HumanMessage(content=f"""
                Tổng hợp từ các nhóm chuyên gia:
                
                {group_summaries_text}
                
                Báo cáo hiện tại:
                {current_report}
                
                Phê bình tổng thể:
                {meta_critique}
                
                Tài liệu đang phân tích có tên: {file_name}
                
                Hãy tinh chỉnh báo cáo cuối cùng dựa trên phê bình tổng thể. Đảm bảo giải quyết các điểm yếu, mâu thuẫn và bổ sung các góc nhìn còn thiếu.
                """)
            ]
            
            # Gọi model
            response = model.invoke(messages)
            refined_report = response.content
            
        except Exception as e:
            print(f"[ERROR] Error in final report refinement: {str(e)}")
            refined_report = current_report  # Giữ nguyên báo cáo cũ nếu có lỗi
        
        # Tạo state mới với báo cáo đã tinh chỉnh
        new_state = cast(Dict[str, Any], {})
        new_state["final_report"] = refined_report
        
        return new_state
    
    return refine_final_report

# Hàm kiểm tra điều kiện dừng vòng lặp
def should_continue_iteration(state: Dict[str, Any]) -> str:
    """
    Kiểm tra xem có nên tiếp tục vòng lặp phê bình và tinh chỉnh hay không.
    
    Args:
        state: Trạng thái hiện tại của hệ thống
        
    Returns:
        "continue" nếu cần tiếp tục vòng lặp, "stop" nếu đủ điều kiện dừng
    """
    # Lấy số lần lặp hiện tại
    current_iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)  # Mặc định tối đa 3 vòng lặp
    
    # Nếu đã đạt số lần lặp tối đa, dừng lại
    if current_iteration >= max_iterations:
        return "stop"
    
    # Có thể thêm logic phức tạp hơn để quyết định dừng sớm dựa trên mức độ cải thiện
    
    # Cập nhật số lần lặp
    state["iteration"] = current_iteration + 1
    
    return "continue"