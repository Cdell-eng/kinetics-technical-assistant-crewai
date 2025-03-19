"""
Agent definitions for the CrewAI integration
"""

from crewai import Agent, Task
from crewai.tools import BaseTool
from langchain.tools import tool
from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field
import os
import re
import logging
import asyncio

logger = logging.getLogger(__name__)

class KineticsSearchTool(BaseTool):
    """Tool for searching Kinetics knowledge base and website."""
    name: str = "KineticsSearch"
    description: str = "Searches the Kinetics knowledge base and website for relevant information"
    
    def __init__(self):
        super().__init__()
    
    @property
    def args_schema(self) -> Type[BaseModel]:
        from pydantic import BaseModel
        
        class SearchInput(BaseModel):
            query: str = Field(description="The search query to look up")
         
        return SearchInput
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        """Execute the search tool asynchronously."""
        # Import here to avoid circular imports
        from ta import search_qdrant, search_kinetics_website, enhance_search_result
        
        try:
            # Run both searches
            kb_results = await search_qdrant(query)
            web_results_raw = await search_kinetics_website(query)
            
            # Enhance web results
            web_results = []
            for result in web_results_raw[:3]:  # Limit to top 3
                enhanced = await enhance_search_result(result, query)
                web_results.append(enhanced)
            
            return {
                "knowledge_base": kb_results,
                "web_results": web_results
            }
        except Exception as e:
            logger.error(f"Error in KineticsSearchTool: {str(e)}")
            return {
                "knowledge_base": [],
                "web_results": [],
                "error": str(e)
            }
            
    def _run(self, query: str) -> Dict[str, Any]:
        """Execute the search tool synchronously."""
        try:
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async function
            result = loop.run_until_complete(self._arun(query))
            
            # Clean up
            loop.close()
            
            return result
        except Exception as e:
            logger.error(f"Error in KineticsSearchTool synchronous execution: {str(e)}")
            return {
                "knowledge_base": [],
                "web_results": [],
                "error": str(e)
            }

class PDFProcessorTool(BaseTool):
    """Tool for processing PDF documents."""
    name: str = "PDFProcessor"
    description: str = "Downloads and extracts text from PDF documents"
    
    def __init__(self):
        super().__init__()
    
    @property
    def args_schema(self) -> Type[BaseModel]:
        class PDFInput(BaseModel):
            url: str = Field(description="The URL of the PDF to process")
            
        return PDFInput
    
    def _execute(self, url: str) -> str:
        """Download and extract text from a PDF document"""
        # Import here to avoid circular imports
        from ta import download_pdf, extract_text_from_pdf
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            pdf_content = loop.run_until_complete(download_pdf(url))
            loop.close()
            
            if not pdf_content:
                return "Failed to download PDF"
            
            text = extract_text_from_pdf(pdf_content)
            return text or "No text content extracted"
        except Exception as e:
            logger.error(f"Error in PDFProcessorTool: {str(e)}")
            return f"Error processing PDF: {str(e)}"

class ProductDatabaseTool(BaseTool):
    """Tool for retrieving product information."""
    name: str = "ProductDatabase"
    description: str = "Retrieves detailed product information from Kinetics database"
    
    def __init__(self):
        super().__init__()
    
    @property
    def args_schema(self) -> Type[BaseModel]:
        class ProductInput(BaseModel):
            product_code: str = Field(description="The product code to look up")
            
        return ProductInput
    
    def _execute(self, product_code: str) -> Dict[str, Any]:
        """Get detailed information about a specific Kinetics product by code"""
        # Extract product code if passed a full description
        product_match = re.search(r'\b(ICW|KSCH|KIP|ESR|RIM|FLM|FIC|HS-\d+)\b', product_code, re.IGNORECASE)
        if product_match:
            product_code = product_match.group(0).upper()
        
        # Import agent memory
        from .memory import AgentMemory
        
        memory = AgentMemory()
        product_info = memory.get_product_info(product_code)
        
        if product_info:
            return {
                "found": True,
                "product_code": product_code,
                "sources": product_info.get("sources", []),
                "mention_count": product_info.get("mention_count", 0),
                "last_mentioned": product_info.get("last_mentioned", "")
            }
        
        # Search Qdrant for this product
        from ta import search_qdrant
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        search_results = loop.run_until_complete(search_qdrant(product_code))
        loop.close()
        
        if search_results:
            # Add to memory
            memory.add_product_reference(
                product_code, 
                search_results[0].get("metadata_storage_name", "Unknown")
            )
            
            return {
                "found": True,
                "product_code": product_code,
                "documents": [
                    {
                        "name": result.get("metadata_storage_name", "Unknown"),
                        "content_preview": result.get("content", "")[:200] + "..."
                    }
                    for result in search_results[:3]
                ],
                "total_results": len(search_results)
            }
        
        return {
            "found": False,
            "product_code": product_code,
            "message": "No information found for this product code"
        }

def select_optimal_model_for_agent(role: str, query: str) -> str:
    """Dynamically select the best model for each agent based on query type"""
    query_lower = query.lower()
    
    if role == "Research Specialist":
        # Claude is good at comprehensive research
        if len(query.split()) > 15 or "detailed" in query_lower:
            return "claude-3-5-sonnet-20241022"
        # GPT is good at quick lookup
        else:
            return "gpt-4o-mini"
            
    elif role == "Technical Document Specialist":
        # GPT is better for structural document analysis
        if any(term in query_lower for term in ["specification", "drawing", "diagram"]):
            return "gpt-4o"
        # Claude is better for nuanced text extraction
        else:
            return "claude-3-5-sonnet-20241022"
            
    elif role == "Kinetics Product Expert":
        # Grok is good for factual synthesis
        if any(term in query_lower for term in ["compare", "difference", "versus"]):
            return "grok-2-latest"
        # Claude is better for nuanced explanations
        else:
            return "claude-3-5-sonnet-20241022"
            
    # Default model selection
    return "claude-3-5-sonnet-20241022"

def create_kinetics_crew(anthropic_api_key, openai_api_key, grok_api_key, query):
    """Create the Kinetics Technical Assistant crew with specialized agents"""
    # Set up API keys
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["GROK_API_KEY"] = grok_api_key
    # Create tools
    kb_search_tool = KineticsSearchTool()
    pdf_processor_tool = PDFProcessorTool()
    product_db_tool = ProductDatabaseTool()
    
    # Determine optimal models for each agent
    research_model = select_optimal_model_for_agent("Research Specialist", query)
    document_model = select_optimal_model_for_agent("Technical Document Specialist", query)
    expert_model = select_optimal_model_for_agent("Kinetics Product Expert", query)
    
    logger.info(f"Selected models - Research: {research_model}, Document: {document_model}, Expert: {expert_model}")
    
    # Create specialized agents
    research_agent = Agent(
        role="Research Specialist",
        goal="Find the most relevant technical information for Kinetics Noise Control products",
        backstory="""You are an expert researcher specializing in acoustic and vibration control. 
        You know how to find the most relevant technical information from both knowledge bases 
        and web resources. Your expertise allows you to quickly identify the most pertinent 
        documents and data for any Kinetics product inquiry.""",
        tools=[kb_search_tool, product_db_tool],
        verbose=True,
        allow_delegation=True,
        model_name=research_model
    )
    
    document_agent = Agent(
        role="Technical Document Specialist",
        goal="Process technical documents and extract relevant information",
        backstory="""You are a technical document specialist with expertise in extracting 
        and summarizing information from PDF documents and technical specifications. You can
        parse complex technical drawings, specifications, and installation guides to identify
        key requirements and specifications. Your strength is in organizing and presenting
        technical information in a clear, structured format.""",
        tools=[pdf_processor_tool],
        verbose=True,
        allow_delegation=True,
        model_name=document_model
    )
    
    expert_agent = Agent(
        role="Kinetics Product Expert",
        goal="Provide accurate technical advice about Kinetics Noise Control products",
        backstory="""You are a senior technical expert with deep knowledge of Kinetics 
        Noise Control products, applications, and installation requirements. You can 
        synthesize information from multiple sources to provide precise answers. You have
        years of experience helping engineers and contractors select and implement the
        right acoustic and vibration solutions for their projects. Your answers are always
        technically accurate and practical.""",
        verbose=True,
        allow_delegation=True,
        model_name=expert_model
    )
    
    # Return the crew and agents for flexible usage
    return {
        "research_agent": research_agent,
        "document_agent": document_agent,
        "expert_agent": expert_agent
    }