"""
Error handling and fallback mechanisms for CrewAI
"""

import logging
import traceback
from functools import wraps
from typing import Callable, Any, Dict, List
import asyncio

logger = logging.getLogger(__name__)

class CrewAIException(Exception):
    """Base exception for CrewAI errors"""
    pass

class AgentExecutionError(CrewAIException):
    """Error during agent execution"""
    pass

class ModelAPIError(CrewAIException):
    """Error connecting to model API"""
    pass

class ContextProcessingError(CrewAIException):
    """Error processing context"""
    pass

class InvalidResponseError(CrewAIException):
    """Error when response is invalid"""
    pass

def with_error_handling(fallback_func: Callable):
    """
    Decorator to add error handling with fallback function
    
    Args:
        fallback_func: Function to call when the main function fails
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                # Call fallback function
                logger.info(f"Using fallback function {fallback_func.__name__}")
                
                try:
                    return await fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Error in fallback function: {str(fallback_error)}")
                    logger.error(traceback.format_exc())
                    
                    # Last resort fallback
                    return {
                        "answer": f"I apologize, but I encountered an error processing your request. Both the primary and fallback systems failed. Please try again or contact support. Primary error: {str(e)}. Fallback error: {str(fallback_error)}",
                        "sources": [],
                        "error": True,
                        "fallback_failed": True
                    }
                    
        return wrapper
    return decorator

async def crew_fallback(question: str, config: Dict) -> Dict:
    """
    Fallback function when CrewAI fails
    
    Args:
        question: The user's question
        config: Configuration dictionary with API keys
        
    Returns:
        Response dictionary with answer and sources
    """
    logger.info(f"Using standard workflow as fallback for: {question}")
    
    # Import here to avoid circular imports
    from ta import get_claude_response, search_qdrant, search_kinetics_website, enhance_search_result, construct_ai_prompt
    
    try:
        # Use the standard workflow
        search_results = await search_qdrant(question)
        web_results = await search_kinetics_website(question)
        
        # Format context from knowledge base
        kb_context = ""
        for i, result in enumerate(search_results):
            doc_name = result.get('metadata_storage_name', 'Unknown document')
            content = result.get('content', '')
            kb_context += f"\n=== DOCUMENT {i+1}: {doc_name} ===\n{content}\n=== END OF DOCUMENT {i+1} ===\n\n"
        
        # Enhance web results
        enhanced_web_results = []
        for result in web_results[:3]:
            enhanced_result = await enhance_search_result(result, question)
            enhanced_web_results.append(enhanced_result)
        
        # Prepare sources
        sources = []
        for result in search_results:
            sources.append({
                "file_name": result.get("metadata_storage_name", "Unknown"),
                "file_path": result.get("metadata_storage_path", ""),
                "context": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                "storage_path": result.get("metadata_storage_path", "")
            })
        
        # Add web sources
        for result in enhanced_web_results:
            sources.append({
                "type": "web",
                "title": result.get("title", "Unknown"),
                "url": result.get("url", ""),
                "context": result.get("page_content", "")[:200] + "..." if len(result.get("page_content", "")) > 200 else result.get("page_content", "")
            })
        
        # Construct prompt and get response
        prompt = construct_ai_prompt(question, kb_context, enhanced_web_results)
        
        # Use Claude if available, otherwise fall back to GPT
        response = None
        try:
            response = await get_claude_response(prompt, "claude-3-5-sonnet-20241022")
        except Exception as claude_error:
            logger.error(f"Claude error in fallback: {str(claude_error)}")
            
            # Try OpenAI as a last resort
            from ta import get_openai_response
            try:
                response = await get_openai_response(prompt, "gpt-4o-mini")
            except Exception as openai_error:
                logger.error(f"OpenAI error in fallback: {str(openai_error)}")
                response = f"I apologize, but I couldn't find a complete answer to your question. Both primary and fallback AI systems encountered errors. Please try rephrasing your question or contact support."
        
        return {
            "answer": response,
            "sources": sources,
            "fallback_used": True
        }
        
    except Exception as e:
        logger.error(f"Error in fallback function: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a basic response as last resort
        return {
            "answer": f"I apologize, but I encountered technical difficulties while processing your request. Please try again later or contact support. Error: {str(e)}",
            "sources": [],
            "error": True
        }

def retry_operation(max_retries=3, delay=2):
    """
    Decorator to retry operations with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__}: {str(e)}")
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed, re-raise the exception
                        raise
                    
                    # Wait with exponential backoff
                    await asyncio.sleep(delay * (2 ** attempt))
            
            # This should never be reached due to the re-raise above
            raise RuntimeError("Unexpected error in retry_operation")
            
        return wrapper
    return decorator