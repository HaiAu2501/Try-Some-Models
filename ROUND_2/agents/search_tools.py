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
    """Input schema for general search query."""
    query: str = Field(..., description="The search query to look up information")

class CompanyQuery(BaseModel):
    """Input schema for company financial data search."""
    company_or_ticker: str = Field(..., description="Company name or stock ticker symbol")
    
class SectorQuery(BaseModel):
    """Input schema for sector performance search."""
    sector: str = Field(..., description="Economic sector (e.g., banking, real estate)")
    market: str = Field(default="Vietnam", description="Market to search in (default: Vietnam)")

class MarketQuery(BaseModel):
    """Input schema for market trends search."""
    market: str = Field(default="Vietnam", description="Market to search for trends (default: Vietnam)")

class EconomicQuery(BaseModel):
    """Input schema for economic indicators search."""
    country: str = Field(default="Vietnam", description="Country to search for economic indicators (default: Vietnam)")

# Khởi tạo DuckDuckGo search utility
search_api = DuckDuckGoSearchAPIWrapper()
# Đặt giá trị max_results mặc định
DEFAULT_MAX_RESULTS = 5

@tool("general_search", args_schema=SearchQuery, return_direct=False)
def general_search(query: str) -> List[Dict[str, str]]:
    """
    Searches for general information on any topic.
    
    Args:
        query: The search query to look up information
        
    Returns:
        List of search results with title, link, and snippet
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
        raise ToolException(f"Error during search: {str(e)}")

@tool("search_financial_data", args_schema=CompanyQuery, return_direct=False)
def search_financial_data(company_or_ticker: str) -> Dict[str, Any]:
    """
    Searches for financial data about a specific company or ticker on Vietnam stock market.
    
    Args:
        company_or_ticker: Company name or stock ticker
        
    Returns:
        Dictionary with financial information
    """
    # Add "financial data" and "Vietnam stock market" to get more relevant results
    base_query = f"{company_or_ticker} financial data Vietnam stock market"
    
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
        raise ToolException(f"Error searching for financial data: {str(e)}")

@tool("search_market_trends", args_schema=MarketQuery, return_direct=False)
def search_market_trends(market: str = "Vietnam") -> Dict[str, Any]:
    """
    Searches for current market trends in a specific market (default: Vietnam).
    
    Args:
        market: Market to search for (default: Vietnam)
        
    Returns:
        Dictionary with market trend information
    """
    # Craft query specific to market trends
    query = f"{market} stock market trends current analysis"
    
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
        raise ToolException(f"Error searching for market trends: {str(e)}")

@tool("search_economic_indicators", args_schema=EconomicQuery, return_direct=False)
def search_economic_indicators(country: str = "Vietnam") -> Dict[str, Any]:
    """
    Searches for current economic indicators for a specific country (default: Vietnam).
    
    Args:
        country: Country to search for economic indicators (default: Vietnam)
        
    Returns:
        Dictionary with economic indicators information
    """
    # Craft query specific to economic indicators
    query = f"{country} economic indicators GDP inflation interest rate current"
    
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
        raise ToolException(f"Error searching for economic indicators: {str(e)}")

@tool("search_sector_performance", args_schema=SectorQuery, return_direct=False)
def search_sector_performance(sector: str, market: str = "Vietnam") -> Dict[str, Any]:
    """
    Searches for performance data about a specific sector in a given market (default: Vietnam).
    
    Args:
        sector: Economic sector (e.g., banking, real estate)
        market: Market to search in (default: Vietnam)
        
    Returns:
        Dictionary with sector performance information
    """
    # Craft query specific to sector performance
    query = f"{sector} sector performance {market} stock market current"
    
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
        raise ToolException(f"Error searching for sector performance: {str(e)}")

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
    Get the list of search tools.
    
    Returns:
        List of search tools
    """
    return search_tools