from typing import List, Dict, Optional, Any
import os
from datetime import datetime
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

# Initialize DuckDuckGo search
search_api = DuckDuckGoSearchAPIWrapper()
DEFAULT_MAX_RESULTS = 5

def simple_search(query: str) -> List[Dict[str, str]]:
    """
    Perform a simple search using DuckDuckGo.
    
    Args:
        query: The search query
        
    Returns:
        List of search results with title, link, and snippet
    """
    try:
        results = search_api.results(query, max_results=DEFAULT_MAX_RESULTS)
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", "")
            })
        
        return formatted_results
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

def search_with_context(query: str, context: str) -> List[Dict[str, str]]:
    """
    Perform a search with additional context.
    
    Args:
        query: The base search query
        context: Additional context to add to the query
        
    Returns:
        List of search results
    """
    enhanced_query = f"{query} {context}"
    return simple_search(enhanced_query)

def search_vietnam_market(query: str) -> List[Dict[str, str]]:
    """
    Perform a search specifically about the Vietnamese market.
    
    Args:
        query: The base search query
        
    Returns:
        List of search results
    """
    return search_with_context(query, "Vietnam stock market")

def search_financial_data(company_or_ticker: str) -> List[Dict[str, str]]:
    """
    Search for financial data about a specific company or ticker.
    
    Args:
        company_or_ticker: Company name or stock ticker
        
    Returns:
        List of search results
    """
    return search_with_context(company_or_ticker, "financial data Vietnam stock market")

def search_sector_performance(sector: str) -> List[Dict[str, str]]:
    """
    Search for performance data about a specific sector in Vietnam.
    
    Args:
        sector: The economic sector (e.g. banking, real estate)
        
    Returns:
        List of search results
    """
    return search_with_context(sector, "sector performance Vietnam stock market")

def search_economic_indicators() -> List[Dict[str, str]]:
    """
    Search for current economic indicators in Vietnam.
    
    Returns:
        List of search results
    """
    return simple_search("Vietnam GDP inflation interest rate economic indicators current")

# Get all search tools for compatibility with the original code
def get_search_tools():
    """
    For compatibility with original code - this function is not used in the new implementation
    but needs to be defined to avoid import errors.
    """
    return []