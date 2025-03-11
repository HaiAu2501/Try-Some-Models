# search_tools.py
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool, tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

# Các model để xác định schema cho các công cụ
class SearchQuery(BaseModel):
    """Schema đầu vào cho truy vấn tìm kiếm tổng quát."""
    query: str = Field(..., description="Truy vấn tìm kiếm để tra cứu thông tin")

class CompanyQuery(BaseModel):
    """Schema đầu vào cho tìm kiếm dữ liệu tài chính công ty."""
    company_or_ticker: str = Field(..., description="Tên công ty hoặc mã cổ phiếu")
    
class SectorQuery(BaseModel):
    """Schema đầu vào cho tìm kiếm hiệu suất ngành."""
    sector: str = Field(..., description="Ngành kinh tế (ví dụ: ngân hàng, bất động sản)")
    market: str = Field(default="Vietnam", description="Thị trường cần tìm kiếm (mặc định: Việt Nam)")

class MarketQuery(BaseModel):
    """Schema đầu vào cho tìm kiếm xu hướng thị trường."""
    market: str = Field(default="Vietnam", description="Thị trường cần tìm kiếm xu hướng (mặc định: Việt Nam)")

class EconomicQuery(BaseModel):
    """Schema đầu vào cho tìm kiếm chỉ số kinh tế."""
    country: str = Field(default="Vietnam", description="Quốc gia cần tìm kiếm chỉ số kinh tế (mặc định: Việt Nam)")

# Khởi tạo DuckDuckGo search utility
search_api = DuckDuckGoSearchAPIWrapper()
# Đặt giá trị max_results mặc định
DEFAULT_MAX_RESULTS = 5

@tool("general_search", args_schema=SearchQuery, return_direct=False)
def general_search(query: str) -> List[Dict[str, str]]:
    """
    Tìm kiếm thông tin tổng quát về bất kỳ chủ đề nào.
    
    Tham số:
        query: Truy vấn tìm kiếm để tra cứu thông tin
        
    Trả về:
        Danh sách kết quả tìm kiếm với tiêu đề, liên kết và đoạn trích
    """
    try:
        # Truyền max_results trực tiếp vào phương thức results()
        results = search_api.results(query, max_results=DEFAULT_MAX_RESULTS)
        
        # Format the results to ensure consistent structure
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", "")
            })
        
        return formatted_results
    except Exception as e:
        raise ToolException(f"Lỗi trong quá trình tìm kiếm: {str(e)}")

@tool("search_financial_data", args_schema=CompanyQuery, return_direct=False)
def search_financial_data(company_or_ticker: str) -> Dict[str, Any]:
    """
    Tìm kiếm dữ liệu tài chính về một công ty hoặc mã cổ phiếu cụ thể trên thị trường chứng khoán Việt Nam.
    
    Tham số:
        company_or_ticker: Tên công ty hoặc mã cổ phiếu
        
    Trả về:
        Dictionary với thông tin tài chính
    """
    # Add "financial data" and "Vietnam stock market" to get more relevant results
    base_query = f"{company_or_ticker} dữ liệu tài chính thị trường chứng khoán Việt Nam"
    
    try:
        # Truyền max_results trực tiếp vào phương thức results()
        results = search_api.results(base_query, max_results=DEFAULT_MAX_RESULTS)
        
        # Extract and format financial data
        financial_data = {
            "company": company_or_ticker,
            "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        
        return financial_data
    except Exception as e:
        raise ToolException(f"Lỗi tìm kiếm dữ liệu tài chính: {str(e)}")

@tool("search_market_trends", args_schema=MarketQuery, return_direct=False)
def search_market_trends(market: str = "Vietnam") -> Dict[str, Any]:
    """
    Tìm kiếm xu hướng thị trường hiện tại trong một thị trường cụ thể (mặc định: Việt Nam).
    
    Tham số:
        market: Thị trường cần tìm kiếm (mặc định: Việt Nam)
        
    Trả về:
        Dictionary với thông tin xu hướng thị trường
    """
    # Craft query specific to market trends
    query = f"{market} xu hướng thị trường chứng khoán phân tích hiện tại"
    
    try:
        # Truyền max_results trực tiếp vào phương thức results()
        results = search_api.results(query, max_results=DEFAULT_MAX_RESULTS)
        
        # Extract and format market trend data
        market_data = {
            "market": market,
            "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        
        return market_data
    except Exception as e:
        raise ToolException(f"Lỗi tìm kiếm xu hướng thị trường: {str(e)}")

@tool("search_economic_indicators", args_schema=EconomicQuery, return_direct=False)
def search_economic_indicators(country: str = "Vietnam") -> Dict[str, Any]:
    """
    Tìm kiếm chỉ số kinh tế hiện tại cho một quốc gia cụ thể (mặc định: Việt Nam).
    
    Tham số:
        country: Quốc gia cần tìm kiếm chỉ số kinh tế (mặc định: Việt Nam)
        
    Trả về:
        Dictionary với thông tin chỉ số kinh tế
    """
    # Craft query specific to economic indicators
    query = f"{country} chỉ số kinh tế GDP lạm phát lãi suất hiện tại"
    
    try:
        # Truyền max_results trực tiếp vào phương thức results()
        results = search_api.results(query, max_results=DEFAULT_MAX_RESULTS)
        
        # Extract and format economic data
        economic_data = {
            "country": country,
            "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        
        return economic_data
    except Exception as e:
        raise ToolException(f"Lỗi tìm kiếm chỉ số kinh tế: {str(e)}")

@tool("search_sector_performance", args_schema=SectorQuery, return_direct=False)
def search_sector_performance(sector: str, market: str = "Vietnam") -> Dict[str, Any]:
    """
    Tìm kiếm dữ liệu hiệu suất về một ngành cụ thể trong một thị trường nhất định (mặc định: Việt Nam).
    
    Tham số:
        sector: Ngành kinh tế (ví dụ: ngân hàng, bất động sản)
        market: Thị trường cần tìm kiếm (mặc định: Việt Nam)
        
    Trả về:
        Dictionary với thông tin hiệu suất ngành
    """
    # Craft query specific to sector performance
    query = f"hiệu suất ngành {sector} thị trường chứng khoán {market} hiện tại"
    
    try:
        # Truyền max_results trực tiếp vào phương thức results()
        results = search_api.results(query, max_results=DEFAULT_MAX_RESULTS)
        
        # Extract and format sector data
        sector_data = {
            "sector": sector,
            "market": market,
            "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        
        return sector_data
    except Exception as e:
        raise ToolException(f"Lỗi tìm kiếm hiệu suất ngành: {str(e)}")

# Tạo danh sách các công cụ tìm kiếm
search_tools = [
    general_search,
    search_financial_data,
    search_market_trends,
    search_economic_indicators,
    search_sector_performance
]

def get_search_tools() -> List[Tool]:
    """
    Lấy danh sách các công cụ tìm kiếm.
    
    Trả về:
        Danh sách các công cụ tìm kiếm
    """
    return search_tools