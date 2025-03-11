# critics.py
from typing import Dict, List, Any, Annotated, TypedDict, cast
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import ToolException
from langchain.prompts import PromptTemplate

from search_tools import get_search_tools

# Lấy công cụ tìm kiếm
search_tools = get_search_tools()

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

Hãy tập trung đánh giá:
- Tính ứng dụng của phân tích đối với chiến lược đầu tư
- Tính đầy đủ và chính xác của dữ liệu được sử dụng
- Mức độ phù hợp của các khuyến nghị với thị trường Việt Nam
- Tính khả thi của các đề xuất đầu tư
- Độ tin cậy và tính cập nhật của thông tin từ các nguồn trực tuyến

Khi đánh giá, hãy phân loại các vấn đề thành:
- Mức độ nghiêm trọng cao: Sai lệch lớn ảnh hưởng đến quyết định đầu tư
- Mức độ trung bình: Thiếu sót đáng kể cần bổ sung
- Mức độ thấp: Có thể cải thiện để hoàn thiện phân tích
</task>
"""

META_CRITIC_PROMPT = """
<role>
Bạn là Tác tử Phê bình Tổng hợp (Meta-Critic Agent) về chiến lược đầu tư.
</role>

<task>
Nhiệm vụ của bạn là:
1. Đánh giá chiến lược đầu tư tổng hợp từ tất cả các nhóm chuyên gia
2. Xác định mâu thuẫn, chồng chéo hoặc thiếu sót giữa các đề xuất
3. Kiểm tra tính nhất quán của chiến lược đầu tư đề xuất
4. Đánh giá hiệu quả của chiến lược quản lý rủi ro
5. Chỉ ra các cải tiến cần thiết cho chiến lược đầu tư cuối cùng
6. Đánh giá việc sử dụng và trích dẫn thông tin từ các nguồn trực tuyến

Hãy tập trung đánh giá:
- Tính khả thi của chiến lược đầu tư trên thị trường Việt Nam
- Mức độ tối ưu của phân bổ tài sản đề xuất
- Tính đầy đủ của các yếu tố vĩ mô và vi mô trong phân tích
- Mức độ phù hợp của các khuyến nghị đầu tư với các loại nhà đầu tư
- Độ tin cậy và tính thời sự của thông tin từ các nguồn trực tuyến

Đưa ra phản hồi cụ thể cho những khía cạnh nào cần được phân tích sâu hơn, và những khuyến nghị nào cần được làm rõ hoặc thêm chi tiết để tạo ra chiến lược đầu tư tối ưu.
</task>
"""

# Hàm để lấy model LLM
def get_model():
    """
    Lấy mô hình LLM phù hợp dựa trên cấu hình môi trường.
    Trả về mô hình OpenAI theo mặc định.
    """
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("Không tìm thấy API key cho OpenAI")

# Tạo tác tử phê bình cho nhóm với tích hợp tìm kiếm
def create_group_critic(group_name: str):
    """
    Tạo tác tử phê bình (critic) cho một nhóm chuyên gia với tích hợp tìm kiếm trực tuyến.
    
    Tham số:
        group_name: Tên của nhóm chuyên gia
        
    Trả về:
        Hàm critic thực hiện đánh giá phân tích của nhóm
    """
    llm = get_model()
    critic_system_prompt = GROUP_CRITIC_PROMPT.format(group_name=group_name)
    
    # Tạo mẫu prompt tác tử
    critic_prompt = PromptTemplate.from_template(
        """
{system_prompt}

Bạn có quyền truy cập vào các công cụ sau để hỗ trợ đánh giá của mình:

{tools}

Sử dụng các công cụ này để nghiên cứu thông tin hiện tại để kiểm tra sự thật và đánh giá các phân tích của chuyên gia.

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
    critic_agent = create_react_agent(llm, search_tools, critic_prompt)
    
    # Tạo trình thực thi tác tử
    critic_executor = AgentExecutor(
        agent=critic_agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=3
    )
    
    def critic_evaluation(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Đánh giá phân tích từ các chuyên gia và tổng hợp của nhóm.
        
        Tham số:
            state: Trạng thái hiện tại của hệ thống
            
        Trả về:
            Trạng thái mới với phản hồi của critic
        """
        try:
            # Trích xuất dữ liệu từ state
            expert_analyses = ""
            group_summary = ""
            
            # Tập hợp các phân tích của chuyên gia trong nhóm
            experts_in_group = []
            for expert, analysis in state.get("analyses", {}).items():
                if expert in state.get("group_experts", {}).get(group_name, []):
                    experts_in_group.append(expert)
                    expert_analyses += f"### Phân tích từ {expert}:\n{analysis}\n\n"
            
            # Lấy tổng hợp của nhóm
            if group_name in state.get("group_summaries", {}):
                group_summary = state["group_summaries"][group_name]
            
            # Trích xuất thông tin tìm kiếm từ các chuyên gia trong nhóm
            search_info = ""
            for expert in experts_in_group:
                search_key = f"{expert}_search"
                if search_key in state.get("search_results", {}):
                    search_data = state["search_results"][search_key]
                    if "intermediate_steps" in search_data:
                        search_info += f"\n### Thông tin tìm kiếm của {expert}:\n"
                        for step in search_data["intermediate_steps"]:
                            if "tool" in step and "input" in step:
                                search_info += f"- Đã sử dụng {step['tool']} để tìm kiếm: {step['input']}\n"
            
            # Lấy thông tin tìm kiếm từ group summarizer
            summarizer_key = f"{group_name}_summarizer_search"
            if summarizer_key in state.get("search_results", {}):
                search_data = state["search_results"][summarizer_key]
                if "intermediate_steps" in search_data:
                    search_info += f"\n### Thông tin tìm kiếm của người tổng hợp nhóm:\n"
                    for step in search_data["intermediate_steps"]:
                        if "tool" in step and "input" in step:
                            search_info += f"- Đã sử dụng {step['tool']} để tìm kiếm: {step['input']}\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running critic for {group_name} on file: {file_name}")
            
            # Tạo đầu vào cho tác tử
            critic_input = f"""
            Hãy đánh giá phê bình (critique) các phân tích sau đây từ nhóm chuyên gia {group_name}:
            
            PHÂN TÍCH CỦA CÁC CHUYÊN GIA:
            {expert_analyses}
            
            TỔNG HỢP CỦA NHÓM:
            {group_summary}
            
            THÔNG TIN TÌM KIẾM ĐÃ THỰC HIỆN:
            {search_info}
            
            Tài liệu đang phân tích có tên: {file_name}
            
            Hãy đưa ra đánh giá chi tiết về:
            1. Độ chính xác của các phân tích (kiểm chứng bằng các tìm kiếm nếu cần)
            2. Tính đầy đủ của phân tích và tổng hợp
            3. Các mâu thuẫn hoặc thiếu sót giữa các chuyên gia
            4. Tính khả thi của các khuyến nghị đầu tư
            5. Mức độ cập nhật của thông tin so với tình hình thị trường hiện tại
            
            Sử dụng các công cụ tìm kiếm để kiểm tra, xác minh hoặc cập nhật thông tin khi cần thiết.
            Phân loại vấn đề theo mức độ nghiêm trọng và đưa ra đề xuất cải thiện cụ thể.
            """
            
            # Thực thi tác tử với công cụ
            critic_result = critic_executor.invoke({
                "system_prompt": critic_system_prompt,
                "input": critic_input,
                "tools": search_tools
            })
            
            # Trích xuất phê bình từ tác tử
            critique = critic_result.get("output", "No critique provided")
            
            # Lưu trữ kết quả tìm kiếm trong trạng thái
            intermediate_steps = critic_result.get("intermediate_steps", [])
            search_results_for_state = {
                f"{group_name}_critic_search": {
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
            print(f"[ERROR] Error in {group_name} critic: {str(e)}")
            critique = f"Error generating critique for {group_name}: {str(e)}"
            search_results_for_state = {
                f"{group_name}_critic_search": {
                    "error": str(e)
                }
            }
        
        # Tạo state mới với phê bình của critic
        new_state = cast(Dict[str, Any], {})
        if "critiques" not in state:
            new_state["critiques"] = {}
        
        new_state["critiques"] = {group_name: critique}
        new_state["search_results"] = search_results_for_state
        
        return new_state
    
    return critic_evaluation

# Tạo tác tử phê bình tổng hợp (meta-critic) với tích hợp tìm kiếm
def create_meta_critic():
    """
    Tạo tác tử phê bình tổng hợp (meta-critic) đánh giá chiến lược đầu tư cuối cùng.
    
    Trả về:
        Hàm meta-critic thực hiện đánh giá tổng thể
    """
    llm = get_model()
    
    # Tạo mẫu prompt tác tử
    meta_critic_prompt = PromptTemplate.from_template(
        """
{system_prompt}

Bạn có quyền truy cập vào các công cụ sau để hỗ trợ đánh giá của mình:

{tools}

Sử dụng các công cụ này để nghiên cứu thông tin hiện tại để đánh giá chiến lược đầu tư.

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
    meta_critic_agent = create_react_agent(llm, search_tools, meta_critic_prompt)
    
    # Tạo trình thực thi tác tử
    meta_critic_executor = AgentExecutor(
        agent=meta_critic_agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5
    )
    
    def meta_critic_evaluation(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Đánh giá chiến lược đầu tư tổng hợp cuối cùng.
        
        Tham số:
            state: Trạng thái hiện tại của hệ thống
            
        Trả về:
            Trạng thái mới với phản hồi của meta-critic
        """
        try:
            # Trích xuất dữ liệu từ state
            group_summaries_text = ""
            final_report = state.get("final_report", "")
            
            # Tập hợp các tổng hợp từ các nhóm
            for group_name, summary in state.get("group_summaries", {}).items():
                group_summaries_text += f"### Tổng kết từ nhóm {group_name}:\n{summary}\n\n"
            
            # Lấy thông tin tìm kiếm từ final synthesizer
            synthesizer_search = ""
            if "final_synthesizer_search" in state.get("search_results", {}):
                search_data = state["search_results"]["final_synthesizer_search"]
                if "intermediate_steps" in search_data:
                    synthesizer_search += f"\n### Thông tin tìm kiếm của final synthesizer:\n"
                    for step in search_data["intermediate_steps"]:
                        if "tool" in step and "input" in step:
                            synthesizer_search += f"- Đã sử dụng {step['tool']} để tìm kiếm: {step['input']}\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Running meta-critic on file: {file_name}")
            
            # Tạo đầu vào cho tác tử
            meta_critic_input = f"""
            Hãy đánh giá phê bình (critique) chiến lược đầu tư tổng hợp sau đây:
            
            TỔNG KẾT TỪ CÁC NHÓM CHUYÊN GIA:
            {group_summaries_text}
            
            CHIẾN LƯỢC ĐẦU TƯ CUỐI CÙNG:
            {final_report}
            
            THÔNG TIN TÌM KIẾM ĐÃ THỰC HIỆN:
            {synthesizer_search}
            
            Tài liệu đang phân tích có tên: {file_name}
            
            Hãy đưa ra đánh giá chi tiết về:
            1. Tính khả thi của chiến lược đầu tư trên thị trường Việt Nam
            2. Mức độ đầy đủ và nhất quán trong phân tích
            3. Độ phù hợp của phân bổ tài sản đề xuất
            4. Hiệu quả của chiến lược quản lý rủi ro
            5. Tính cập nhật của thông tin so với điều kiện thị trường hiện tại
            
            Sử dụng các công cụ tìm kiếm để kiểm tra, xác minh hoặc cập nhật thông tin khi cần thiết,
            đặc biệt là thông tin về các điều kiện thị trường mới nhất và các quy định liên quan.
            
            Đưa ra những đề xuất cụ thể để cải thiện chiến lược đầu tư và làm cho nó phù hợp hơn
            với thị trường Việt Nam hiện tại.
            """
            
            # Thực thi tác tử với công cụ
            meta_critic_result = meta_critic_executor.invoke({
                "system_prompt": META_CRITIC_PROMPT,
                "input": meta_critic_input,
                "tools": search_tools
            })
            
            # Trích xuất phê bình tổng thể từ tác tử
            meta_critique = meta_critic_result.get("output", "No meta critique provided")
            
            # Lưu trữ kết quả tìm kiếm trong trạng thái
            intermediate_steps = meta_critic_result.get("intermediate_steps", [])
            search_results_for_state = {
                "meta_critic_search": {
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
            print(f"[ERROR] Error in meta-critic: {str(e)}")
            meta_critique = f"Error generating meta-critique: {str(e)}"
            search_results_for_state = {
                "meta_critic_search": {
                    "error": str(e)
                }
            }
        
        # Tạo state mới với phê bình của meta-critic
        new_state = cast(Dict[str, Any], {})
        new_state["meta_critique"] = meta_critique
        new_state["search_results"] = search_results_for_state
        
        return new_state
    
    return meta_critic_evaluation

# Hàm tinh chỉnh phân tích dựa trên phê bình
def create_refinement_agent(expert_name: str, expert_system_prompt: str):
    """
    Tạo tác tử tinh chỉnh phân tích dựa trên phê bình với tích hợp tìm kiếm.
    
    Tham số:
        expert_name: Tên của chuyên gia
        expert_system_prompt: System prompt gốc của chuyên gia
        
    Trả về:
        Hàm refinement agent thực hiện tinh chỉnh phân tích
    """
    llm = get_model()
    
    # Tạo mẫu prompt tác tử
    refinement_prompt = PromptTemplate.from_template(
        """
{system_prompt}

Bạn có quyền truy cập vào các công cụ sau để hỗ trợ phân tích của mình:

{tools}

Sử dụng các công cụ này để nghiên cứu thông tin hiện tại để cải thiện phân tích của bạn dựa trên phê bình.

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
    refinement_agent = create_react_agent(llm, search_tools, refinement_prompt)
    
    # Tạo trình thực thi tác tử
    refinement_executor = AgentExecutor(
        agent=refinement_agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=3
    )
    
    def refine_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tinh chỉnh phân tích dựa trên phê bình.
        
        Tham số:
            state: Trạng thái hiện tại của hệ thống
            
        Trả về:
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
            
            # Lấy thông tin tìm kiếm trước đó của chuyên gia
            search_info = ""
            search_key = f"{expert_name}_search"
            if search_key in state.get("search_results", {}):
                search_data = state["search_results"][search_key]
                if "intermediate_steps" in search_data:
                    search_info += f"\n### Thông tin tìm kiếm trước đó:\n"
                    for step in search_data["intermediate_steps"]:
                        if "tool" in step and "input" in step:
                            search_info += f"- Đã sử dụng {step['tool']} để tìm kiếm: {step['input']}\n"
            
            print(f"\n[DEBUG] Refining analysis for {expert_name} on file: {file_name}")
            
            # Nếu không có phê bình, giữ nguyên phân tích
            if not critiques:
                return {"analyses": {expert_name: current_analysis}}
            
            # Tạo system prompt cho tinh chỉnh
            refinement_system_prompt = f"""
            {expert_system_prompt}
            
            Bạn cần tinh chỉnh phân tích trước đó dựa trên phê bình nhận được và thông tin mới nhất.
            Hãy giải quyết các vấn đề được nêu ra trong phê bình và cải thiện chất lượng phân tích.
            Tập trung vào mục tiêu tối ưu hóa chiến lược đầu tư trên thị trường Việt Nam.
            Khi sử dụng thông tin mới, hãy trích dẫn nguồn và đánh giá độ tin cậy của thông tin.
            """
            
            # Tạo đầu vào cho tác tử
            refinement_input = f"""
            Hãy tinh chỉnh phân tích sau đây dựa trên các phê bình nhận được:
            
            DỮ LIỆU CẦN PHÂN TÍCH:
            {input_data}
            
            PHÂN TÍCH HIỆN TẠI CỦA BẠN:
            {current_analysis}
            
            PHÊ BÌNH NHẬN ĐƯỢC:
            {critiques}
            
            THÔNG TIN TÌM KIẾM TRƯỚC ĐÓ:
            {search_info}
            
            Tài liệu có tên: {file_name}
            
            Hãy sử dụng các công cụ tìm kiếm để khắc phục các thiếu sót được chỉ ra trong phê bình,
            cập nhật thông tin, và cải thiện phân tích của bạn. Đảm bảo rằng phân tích mới:
            
            1. Giải quyết tất cả các vấn đề được nêu trong phê bình
            2. Cập nhật với thông tin thị trường mới nhất
            3. Cung cấp khuyến nghị đầu tư cụ thể và khả thi
            4. Trích dẫn nguồn cho thông tin bên ngoài
            
            Đưa ra phân tích hoàn chỉnh đã được cải thiện.
            """
            
            # Thực thi tác tử với công cụ
            refinement_result = refinement_executor.invoke({
                "system_prompt": refinement_system_prompt,
                "input": refinement_input,
                "tools": search_tools
            })
            
            # Trích xuất phân tích đã tinh chỉnh từ tác tử
            refined_analysis = refinement_result.get("output", "No analysis provided")
            
            # Lưu trữ kết quả tìm kiếm trong trạng thái
            intermediate_steps = refinement_result.get("intermediate_steps", [])
            search_results_for_state = {
                f"{expert_name}_refinement_search": {
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
            print(f"[ERROR] Error in refinement for {expert_name}: {str(e)}")
            refined_analysis = current_analysis  # Giữ nguyên phân tích cũ nếu có lỗi
            search_results_for_state = {
                f"{expert_name}_refinement_search": {
                    "error": str(e)
                }
            }
        
        # Tạo state mới với phân tích đã tinh chỉnh
        new_state = cast(Dict[str, Any], {})
        new_state["analyses"] = {expert_name: refined_analysis}
        new_state["search_results"] = search_results_for_state
        
        return new_state
    
    return refine_analysis

# Hàm tinh chỉnh tổng hợp nhóm dựa trên phê bình
def create_group_summary_refinement(group_name: str):
    """
    Tạo tác tử tinh chỉnh tổng hợp nhóm dựa trên phê bình với tích hợp tìm kiếm.
    
    Tham số:
        group_name: Tên của nhóm
        
    Trả về:
        Hàm refinement agent thực hiện tinh chỉnh tổng hợp nhóm
    """
    llm = get_model()
    
    # Tạo mẫu prompt tác tử
    summary_refinement_prompt = PromptTemplate.from_template(
        """
{system_prompt}

Bạn có quyền truy cập vào các công cụ sau để hỗ trợ tổng hợp của mình:

{tools}

Sử dụng các công cụ này để nghiên cứu thông tin hiện tại để cải thiện bản tổng hợp nhóm của bạn dựa trên phê bình.

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
    summary_refinement_agent = create_react_agent(llm, search_tools, summary_refinement_prompt)
    
    # Tạo trình thực thi tác tử
    summary_refinement_executor = AgentExecutor(
        agent=summary_refinement_agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=3
    )
    
    def refine_group_summary(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tinh chỉnh tổng hợp nhóm dựa trên phê bình.
        
        Tham số:
            state: Trạng thái hiện tại của hệ thống
            
        Trả về:
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
            
            # Lấy thông tin tìm kiếm từ group summarizer
            search_info = ""
            search_key = f"{group_name}_summarizer_search"
            if search_key in state.get("search_results", {}):
                search_data = state["search_results"][search_key]
                if "intermediate_steps" in search_data:
                    search_info += f"\n### Thông tin tìm kiếm trước đó:\n"
                    for step in search_data["intermediate_steps"]:
                        if "tool" in step and "input" in step:
                            search_info += f"- Đã sử dụng {step['tool']} để tìm kiếm: {step['input']}\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Refining summary for group {group_name} on file: {file_name}")
            
            # Tạo system prompt cho tinh chỉnh
            summary_system_prompt = f"""
            Bạn là người tổng hợp ý kiến cho nhóm chuyên gia {group_name}.
            Nhiệm vụ của bạn là tinh chỉnh tổng hợp trước đó dựa trên phê bình nhận được và thông tin cập nhật.
            Hãy giải quyết các vấn đề được nêu ra trong phê bình và cải thiện chất lượng tổng hợp.
            Tập trung vào mục tiêu tối ưu hóa chiến lược đầu tư trên thị trường Việt Nam.
            Khi sử dụng thông tin mới, hãy trích dẫn nguồn và đánh giá độ tin cậy của thông tin.
            """
            
            # Tạo đầu vào cho tác tử
            summary_input = f"""
            Hãy tinh chỉnh tổng hợp nhóm sau đây dựa trên phê bình nhận được:
            
            PHÂN TÍCH CỦA CÁC CHUYÊN GIA TRONG NHÓM:
            {expert_analyses}
            
            TỔNG HỢP HIỆN TẠI CỦA NHÓM:
            {current_summary}
            
            PHÊ BÌNH NHẬN ĐƯỢC:
            {group_critique}
            
            THÔNG TIN TÌM KIẾM TRƯỚC ĐÓ:
            {search_info}
            
            Tài liệu đang phân tích có tên: {file_name}
            
            Hãy sử dụng các công cụ tìm kiếm để khắc phục các thiếu sót được chỉ ra trong phê bình,
            cập nhật thông tin, và cải thiện tổng hợp của nhóm. Đảm bảo rằng tổng hợp mới:
            
            1. Giải quyết tất cả các vấn đề được nêu trong phê bình
            2. Cập nhật với thông tin thị trường mới nhất
            3. Cung cấp khuyến nghị đầu tư cụ thể và khả thi
            4. Tích hợp tốt các góc nhìn khác nhau từ các chuyên gia
            5. Trích dẫn nguồn cho thông tin bên ngoài
            
            Đưa ra tổng hợp hoàn chỉnh đã được cải thiện.
            """
            
            # Thực thi tác tử với công cụ
            summary_result = summary_refinement_executor.invoke({
                "system_prompt": summary_system_prompt,
                "input": summary_input,
                "tools": search_tools
            })
            
            # Trích xuất tổng hợp đã tinh chỉnh từ tác tử
            refined_summary = summary_result.get("output", "No summary provided")
            
            # Lưu trữ kết quả tìm kiếm trong trạng thái
            intermediate_steps = summary_result.get("intermediate_steps", [])
            search_results_for_state = {
                f"{group_name}_summary_refinement_search": {
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
            print(f"[ERROR] Error in summary refinement for {group_name}: {str(e)}")
            refined_summary = current_summary  # Giữ nguyên tổng hợp cũ nếu có lỗi
            search_results_for_state = {
                f"{group_name}_summary_refinement_search": {
                    "error": str(e)
                }
            }
        
        # Tạo state mới với tổng hợp đã tinh chỉnh
        new_state = cast(Dict[str, Any], {})
        new_state["group_summaries"] = {group_name: refined_summary}
        new_state["search_results"] = search_results_for_state
        
        return new_state
    
    return refine_group_summary

# Hàm tinh chỉnh báo cáo cuối cùng dựa trên phê bình tổng thể
def create_final_report_refinement():
    """
    Tạo tác tử tinh chỉnh chiến lược đầu tư cuối cùng dựa trên phê bình tổng thể với tích hợp tìm kiếm.
    
    Trả về:
        Hàm refinement agent thực hiện tinh chỉnh báo cáo cuối cùng
    """
    llm = get_model()
    
    # Tạo mẫu prompt tác tử
    report_refinement_prompt = PromptTemplate.from_template(
        """
{system_prompt}

Bạn có quyền truy cập vào các công cụ sau để hỗ trợ tinh chỉnh của mình:

{tools}

Sử dụng các công cụ này để nghiên cứu thông tin hiện tại để cải thiện chiến lược đầu tư cuối cùng dựa trên phê bình.

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
    report_refinement_agent = create_react_agent(llm, search_tools, report_refinement_prompt)
    
    # Tạo trình thực thi tác tử
    report_refinement_executor = AgentExecutor(
        agent=report_refinement_agent,
        tools=search_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5
    )
    
    def refine_final_report(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tinh chỉnh chiến lược đầu tư cuối cùng dựa trên phê bình tổng thể.
        
        Tham số:
            state: Trạng thái hiện tại của hệ thống
            
        Trả về:
            Trạng thái mới với chiến lược đầu tư đã tinh chỉnh
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
            
            # Lấy thông tin tìm kiếm từ final synthesizer
            search_info = ""
            if "final_synthesizer_search" in state.get("search_results", {}):
                search_data = state["search_results"]["final_synthesizer_search"]
                if "intermediate_steps" in search_data:
                    search_info += f"\n### Thông tin tìm kiếm trước đó:\n"
                    for step in search_data["intermediate_steps"]:
                        if "tool" in step and "input" in step:
                            search_info += f"- Đã sử dụng {step['tool']} để tìm kiếm: {step['input']}\n"
            
            # Lấy thông tin tìm kiếm từ meta critic
            if "meta_critic_search" in state.get("search_results", {}):
                search_data = state["search_results"]["meta_critic_search"]
                if "intermediate_steps" in search_data:
                    search_info += f"\n### Thông tin tìm kiếm của meta critic:\n"
                    for step in search_data["intermediate_steps"]:
                        if "tool" in step and "input" in step:
                            search_info += f"- Đã sử dụng {step['tool']} để tìm kiếm: {step['input']}\n"
            
            file_name = state.get("file_name", "Unknown file")
            
            print(f"\n[DEBUG] Refining final investment strategy on file: {file_name}")
            
            # Tạo system prompt cho tinh chỉnh
            report_system_prompt = """
            Bạn là chuyên gia tổng hợp chiến lược đầu tư cuối cùng.
            Nhiệm vụ của bạn là tinh chỉnh chiến lược đầu tư trước đó dựa trên phê bình tổng thể nhận được và thông tin mới nhất.
            Hãy giải quyết các vấn đề được nêu ra trong phê bình và cải thiện chất lượng chiến lược đầu tư.
            Tập trung vào việc tạo ra chiến lược đầu tư tối ưu, khả thi và phù hợp với thị trường Việt Nam.
            Khi sử dụng thông tin mới, hãy trích dẫn nguồn và đánh giá độ tin cậy của thông tin.
            """
            
            # Tạo đầu vào cho tác tử
            report_input = f"""
            Hãy tinh chỉnh chiến lược đầu tư sau đây dựa trên phê bình tổng thể nhận được:
            
            TỔNG HỢP TỪ CÁC NHÓM CHUYÊN GIA:
            {group_summaries_text}
            
            CHIẾN LƯỢC ĐẦU TƯ HIỆN TẠI:
            {current_report}
            
            PHÊ BÌNH TỔNG THỂ:
            {meta_critique}
            
            THÔNG TIN TÌM KIẾM TRƯỚC ĐÓ:
            {search_info}
            
            Tài liệu đang phân tích có tên: {file_name}
            
            Hãy sử dụng các công cụ tìm kiếm để khắc phục các thiếu sót được chỉ ra trong phê bình,
            cập nhật thông tin, và cải thiện chiến lược đầu tư. Đảm bảo rằng chiến lược mới:
            
            1. Giải quyết tất cả các vấn đề được nêu trong phê bình tổng thể
            2. Cập nhật với thông tin thị trường mới nhất
            3. Cung cấp phân bổ tài sản cụ thể
            4. Đề xuất các ngành và cổ phiếu tiềm năng
            5. Chỉ rõ thời điểm tham gia thị trường
            6. Đưa ra kế hoạch quản lý rủi ro chi tiết
            7. Cung cấp các hành động cụ thể cho nhà đầu tư
            
            Đưa ra chiến lược đầu tư hoàn chỉnh đã được cải thiện.
            """
            
            # Thực thi tác tử với công cụ
            report_result = report_refinement_executor.invoke({
                "system_prompt": report_system_prompt,
                "input": report_input,
                "tools": search_tools
            })
            
            # Trích xuất báo cáo đã tinh chỉnh từ tác tử
            refined_report = report_result.get("output", "No report provided")
            
            # Lưu trữ kết quả tìm kiếm trong trạng thái
            intermediate_steps = report_result.get("intermediate_steps", [])
            search_results_for_state = {
                "final_report_refinement_search": {
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
            print(f"[ERROR] Error in final report refinement: {str(e)}")
            refined_report = current_report  # Giữ nguyên báo cáo cũ nếu có lỗi
            search_results_for_state = {
                "final_report_refinement_search": {
                    "error": str(e)
                }
            }
        
        # Tạo state mới với báo cáo đã tinh chỉnh
        new_state = cast(Dict[str, Any], {})
        new_state["final_report"] = refined_report
        new_state["search_results"] = search_results_for_state
        
        return new_state
    
    return refine_final_report

# Hàm kiểm tra điều kiện dừng vòng lặp
def should_continue_iteration(state: Dict[str, Any]) -> str:
    """
    Kiểm tra xem có nên tiếp tục vòng lặp phê bình và tinh chỉnh hay không.
    
    Tham số:
        state: Trạng thái hiện tại của hệ thống
        
    Trả về:
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