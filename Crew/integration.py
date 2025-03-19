"""
Integration module for connecting CrewAI with the Kinetics Technical Assistant
"""

from crewai import Crew, Process
import asyncio
import time
from typing import Dict, List, Any
import logging
import traceback

from .agents import create_kinetics_crew
from .tasks import select_appropriate_crew
from .caching import ResponseCache
from .memory import AgentMemory
from .feedback import AgentFeedback
from .error_handlers import with_error_handling, crew_fallback

logger = logging.getLogger(__name__)

# Initialize cache and memory
response_cache = ResponseCache()
agent_memory = AgentMemory()

@with_error_handling(crew_fallback)
async def get_crew_response(question: str, config: Dict) -> Dict:
    """Get a response using CrewAI workflow with caching and error handling"""
    start_time = time.time()
    
    # Log the question
    logger.info(f"Processing CrewAI query: {question[:100]}...")
    
    # Add to agent memory
    agent_memory.add_question(question)
    
    # Check cache first
    cached_response = response_cache.get(question)
    if cached_response:
        logger.info(f"Using cached response for: {question[:50]}...")
        cached_response["from_cache"] = True
        # Calculate response time from cache
        response_time = time.time() - start_time
        
        # Update performance metrics
        try:
            monitor = AgentFeedback()
            monitor.add_feedback(
                query=question,
                agent="CrewAI (cached)",
                feedback={
                    "response_quality": 5,  # Assume high quality for cached responses
                    "information_accuracy": 5,
                    "sources_used": len(cached_response.get("sources", [])),
                    "user_satisfaction": 5,
                    "areas_for_improvement": "Using cached response",
                    "suggested_improvement": ""
                }
            )
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
        
        return cached_response
    
    try:
        # Create the agents
        agents = create_kinetics_crew(
            anthropic_api_key=config["ANTHROPIC_API_KEY"],
            openai_api_key=config["OPENAI_API_KEY"],
            grok_api_key=config.get("GROK_API_KEY", ""),
            query=question
        )
        
        # Select appropriate crew configuration based on query type
        tasks = select_appropriate_crew(question, agents)
        
        # Create the crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=True,
            process=Process.sequential  # Use sequential processing for most reliable results
        )
        
        # Run the crew
        logger.info(f"Starting CrewAI run for: {question[:50]}...")
        result = crew.kickoff()
        logger.info(f"CrewAI run completed for: {question[:50]}...")
        
        # Process sources and create response object
        sources = []
        agent_iterations = 0
        
        # Extract source information from tasks
        for task in tasks:
            if hasattr(task, 'output'):
                # Count iterations
                if hasattr(task, 'iterations'):
                    agent_iterations += task.iterations
                
                # Extract knowledge base sources if available in the output
                if isinstance(task.output, dict) and "knowledge_base" in task.output:
                    kb_sources = task.output.get("knowledge_base", [])
                    
                    for source in kb_sources:
                        if isinstance(source, dict):
                            sources.append({
                                "file_name": source.get("metadata_storage_name", "Unknown"),
                                "file_path": source.get("metadata_storage_path", ""),
                                "context": source.get("content", "")[:200] + "..." if len(source.get("content", "")) > 200 else source.get("content", ""),
                                "storage_path": source.get("metadata_storage_path", "")
                            })
                
                # Extract web sources if available in the output
                if isinstance(task.output, dict) and "web_results" in task.output:
                    web_sources = task.output.get("web_results", [])
                    
                    for source in web_sources:
                        if isinstance(source, dict):
                            sources.append({
                                "type": "web",
                                "title": source.get("title", "Unknown"),
                                "url": source.get("url", ""),
                                "context": source.get("page_content", "")[:200] + "..." if len(source.get("page_content", "")) > 200 else source.get("page_content", "")
                            })
                            
                            # Also process any PDF links from web results
                            if "pdf_links" in source and isinstance(source["pdf_links"], list):
                                for pdf in source["pdf_links"]:
                                    if isinstance(pdf, dict):
                                        sources.append({
                                            "type": "pdf",
                                            "title": pdf.get("title", "PDF Document"),
                                            "url": pdf.get("url", ""),
                                            "context": pdf.get("context", "")
                                        })
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Create response object
        response = {
            "answer": result,
            "sources": sources,
            "from_cache": False,
            "response_time": response_time,
            "agent_iterations": agent_iterations
        }
        
        # Cache the response
        response_cache.set(question, response)
        
        # Log performance metrics
        try:
            monitor = AgentFeedback()
            monitor.add_feedback(
                query=question,
                agent="CrewAI",
                feedback={
                    "response_quality": 4,  # Default quality score
                    "information_accuracy": 4,
                    "sources_used": len(sources),
                    "user_satisfaction": 4,
                    "areas_for_improvement": "",
                    "suggested_improvement": ""
                }
            )
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
        
        logger.info(f"CrewAI response generated in {response_time:.2f}s with {len(sources)} sources")
        return response
        
    except Exception as e:
        logger.error(f"Error in CrewAI workflow: {str(e)}")
        logger.error(traceback.format_exc())
        
        # This will be caught by the error_handling decorator and routed to the fallback
        raise