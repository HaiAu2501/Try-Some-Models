# search_utils.py
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

class EnhancedDuckDuckGoSearch:
    """
    Wrapper class for DuckDuckGo search with enhanced features for financial and market data.
    """
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the search utility.
        
        Args:
            max_results: Maximum number of search results to return
        """
        self.search_api = DuckDuckGoSearchAPIWrapper(max_results=max_results)
        
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a general search using DuckDuckGo.
        
        Args:
            query: Search query
            
        Returns:
            List of search results with title, link, and snippet
        """
        try:
            results = self.search_api.results(query)
            
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
            print(f"[ERROR] Search error: {str(e)}")
            return []
    
    def search_financial_data(self, company_or_ticker: str) -> Dict[str, Any]:
        """
        Search for financial data about a specific company or ticker.
        
        Args:
            company_or_ticker: Company name or stock ticker
            
        Returns:
            Dictionary with financial information
        """
        # Add "financial data" and "Vietnam stock market" to get more relevant results
        base_query = f"{company_or_ticker} financial data Vietnam stock market"
        
        try:
            results = self.search_api.results(base_query)
            
            # Extract and format financial data
            financial_data = {
                "company": company_or_ticker,
                "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results
            }
            
            return financial_data
        except Exception as e:
            print(f"[ERROR] Financial data search error: {str(e)}")
            return {
                "company": company_or_ticker,
                "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": [],
                "error": str(e)
            }
    
    def search_market_trends(self, market: str = "Vietnam") -> Dict[str, Any]:
        """
        Search for current market trends.
        
        Args:
            market: Market to search for (default: Vietnam)
            
        Returns:
            Dictionary with market trend information
        """
        # Craft query specific to market trends
        query = f"{market} stock market trends current analysis"
        
        try:
            results = self.search_api.results(query)
            
            # Extract and format market trend data
            market_data = {
                "market": market,
                "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results
            }
            
            return market_data
        except Exception as e:
            print(f"[ERROR] Market trends search error: {str(e)}")
            return {
                "market": market,
                "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": [],
                "error": str(e)
            }
    
    def search_economic_indicators(self, country: str = "Vietnam") -> Dict[str, Any]:
        """
        Search for current economic indicators for a specific country.
        
        Args:
            country: Country to search for (default: Vietnam)
            
        Returns:
            Dictionary with economic indicators information
        """
        # Craft query specific to economic indicators
        query = f"{country} economic indicators GDP inflation interest rate current"
        
        try:
            results = self.search_api.results(query)
            
            # Extract and format economic data
            economic_data = {
                "country": country,
                "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results
            }
            
            return economic_data
        except Exception as e:
            print(f"[ERROR] Economic indicators search error: {str(e)}")
            return {
                "country": country,
                "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": [],
                "error": str(e)
            }
    
    def search_sector_performance(self, sector: str, market: str = "Vietnam") -> Dict[str, Any]:
        """
        Search for performance data about a specific sector.
        
        Args:
            sector: Economic sector (e.g., banking, real estate)
            market: Market to search in (default: Vietnam)
            
        Returns:
            Dictionary with sector performance information
        """
        # Craft query specific to sector performance
        query = f"{sector} sector performance {market} stock market current"
        
        try:
            results = self.search_api.results(query)
            
            # Extract and format sector data
            sector_data = {
                "sector": sector,
                "market": market,
                "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": results
            }
            
            return sector_data
        except Exception as e:
            print(f"[ERROR] Sector performance search error: {str(e)}")
            return {
                "sector": sector,
                "market": market,
                "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": [],
                "error": str(e)
            }

# Create a singleton instance for use throughout the application
search_tool = EnhancedDuckDuckGoSearch(max_results=5)

def get_search_tool() -> EnhancedDuckDuckGoSearch:
    """
    Get the singleton instance of the search tool.
    
    Returns:
        EnhancedDuckDuckGoSearch instance
    """
    return search_tool