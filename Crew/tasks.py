"""
Task definitions for the CrewAI integration
"""

from crewai import Task
from typing import Dict, List, Any
import re
import logging

logger = logging.getLogger(__name__)

def create_kinetics_tasks(agents, query):
    """Create standard tasks for the Kinetics crew"""
    research_task = Task(
        description=f"Research comprehensive information about: {query}",
        expected_output="A detailed research report with knowledge base findings and web search results",
        agent=agents["research_agent"]
    )
    
    document_processing_task = Task(
        description="Process any PDF documents found during research to extract relevant technical details",
        expected_output="Extracted and summarized information from technical documents",
        agent=agents["document_agent"],
        context=[research_task]  # This task depends on research_task results
    )
    
    expert_analysis_task = Task(
        description=f"Analyze all information and provide a comprehensive answer to: {query}",
        expected_output="A complete technical answer that synthesizes all available information",
        agent=agents["expert_agent"],
        context=[research_task, document_processing_task]  # This task depends on both previous tasks
    )
    
    return [research_task, document_processing_task, expert_analysis_task]

def create_technical_specification_crew(agents, query):
    """Create a crew specialized for technical specification questions"""
    research_task = Task(
        description=f"Research detailed technical specifications for: {query}",
        expected_output="Technical specifications including dimensions, materials, performance metrics, and test results",
        agent=agents["research_agent"]
    )
    
    document_task = Task(
        description="Extract and organize detailed specifications from technical documents",
        expected_output="Organized technical specifications with references to source documents",
        agent=agents["document_agent"],
        context=[research_task]
    )
    
    comparison_task = Task(
        description="Compare specifications with industry standards and competing products",
        expected_output="Comparative analysis highlighting advantages and limitations",
        agent=agents["expert_agent"],
        context=[research_task, document_task]
    )
    
    return [research_task, document_task, comparison_task]

def create_installation_crew(agents, query):
    """Create a crew specialized for installation questions"""
    research_task = Task(
        description=f"Find installation guides and procedures for: {query}",
        expected_output="Installation requirements, steps, and best practices",
        agent=agents["research_agent"]
    )
    
    document_task = Task(
        description="Extract specific installation instructions from technical documents",
        expected_output="Step-by-step installation guide with diagrams if available",
        agent=agents["document_agent"],
        context=[research_task]
    )
    
    expert_task = Task(
        description="Provide expert installation advice and common pitfalls to avoid",
        expected_output="Expert installation guidance with troubleshooting tips",
        agent=agents["expert_agent"],
        context=[research_task, document_task]
    )
    
    return [research_task, document_task, expert_task]

def create_product_selection_crew(agents, query):
    """Create a crew specialized for product selection questions"""
    research_task = Task(
        description=f"Research available products that match the requirements in: {query}",
        expected_output="A list of suitable Kinetics products with key specifications",
        agent=agents["research_agent"]
    )
    
    comparison_task = Task(
        description="Compare and contrast suitable products for this application",
        expected_output="Detailed comparison of product options with pros and cons",
        agent=agents["document_agent"],
        context=[research_task]
    )
    
    recommendation_task = Task(
        description="Recommend the optimal product selection with justification",
        expected_output="Product recommendation with technical justification",
        agent=agents["expert_agent"],
        context=[research_task, comparison_task]
    )
    
    return [research_task, comparison_task, recommendation_task]

def create_troubleshooting_crew(agents, query):
    """Create a crew specialized for troubleshooting questions"""
    research_task = Task(
        description=f"Research common issues and solutions related to: {query}",
        expected_output="Known issues, potential causes, and documented solutions",
        agent=agents["research_agent"]
    )
    
    document_task = Task(
        description="Extract troubleshooting procedures from technical documents",
        expected_output="Detailed troubleshooting steps and maintenance procedures",
        agent=agents["document_agent"],
        context=[research_task]
    )
    
    diagnostic_task = Task(
        description="Diagnose the likely causes and recommend solutions",
        expected_output="Diagnostic analysis with prioritized solution recommendations",
        agent=agents["expert_agent"],
        context=[research_task, document_task]
    )
    
    return [research_task, document_task, diagnostic_task]

def select_appropriate_crew(query: str, agents: Dict) -> List[Task]:
    """Select the appropriate crew configuration based on query content"""
    query_lower = query.lower()
    
    # Check for installation-related queries
    if any(term in query_lower for term in ["install", "mounting", "setup", "assembly", "attach", "secure"]):
        logger.info(f"Selected installation crew for query: {query[:50]}...")
        return create_installation_crew(agents, query)
    
    # Check for specification-related queries
    elif any(term in query_lower for term in ["spec", "dimension", "rating", "performance", "test results", "data"]):
        logger.info(f"Selected technical specification crew for query: {query[:50]}...")
        return create_technical_specification_crew(agents, query)
    
    # Check for product selection queries
    elif any(term in query_lower for term in ["recommend", "best", "which product", "select", "choose", "comparison"]):
        logger.info(f"Selected product selection crew for query: {query[:50]}...")
        return create_product_selection_crew(agents, query)
    
    # Check for troubleshooting queries
    elif any(term in query_lower for term in ["problem", "issue", "not working", "fail", "troubleshoot", "repair"]):
        logger.info(f"Selected troubleshooting crew for query: {query[:50]}...")
        return create_troubleshooting_crew(agents, query)
    
    # Default to standard crew configuration
    logger.info(f"Selected standard crew for query: {query[:50]}...")
    return create_kinetics_tasks(agents, query)