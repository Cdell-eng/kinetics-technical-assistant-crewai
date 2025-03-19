"""
Feedback system for CrewAI agents to gather performance metrics
"""

from typing import Dict, List, Any
import pandas as pd
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AgentFeedback:
    def __init__(self, feedback_file="G:/APPS/Technical Assistant/agent_feedback.xlsx"):
        """
        Initialize the agent feedback system
        
        Args:
            feedback_file: Path to the feedback Excel file
        """
        self.feedback_file = feedback_file
        self._ensure_feedback_file()
    
    def _ensure_feedback_file(self):
        """Create feedback file if it doesn't exist"""
        if not os.path.exists(self.feedback_file):
            try:
                # Create DataFrame with columns
                df = pd.DataFrame(columns=[
                    'Timestamp',
                    'Query',
                    'Agent',
                    'Response Quality',
                    'Information Accuracy',
                    'Sources Used',
                    'User Satisfaction',
                    'Response Time',
                    'Areas for Improvement',
                    'Suggested Improvement'
                ])
                
                # Create directory if needed
                os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
                
                # Save to Excel
                df.to_excel(self.feedback_file, index=False)
                
                logger.info(f"Created new feedback file: {self.feedback_file}")
                
            except Exception as e:
                logger.error(f"Error creating feedback file: {e}")
    
    def add_feedback(self, query: str, agent: str, feedback: Dict[str, Any]) -> bool:
        """
        Add feedback for an agent response
        
        Args:
            query: The user query
            agent: The agent name or type
            feedback: Dictionary of feedback metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read existing data
            df = pd.read_excel(self.feedback_file)
            
            # Create new entry
            new_entry = {
                'Timestamp': datetime.now().isoformat(),
                'Query': query,
                'Agent': agent,
                'Response Quality': feedback.get('response_quality', 0),
                'Information Accuracy': feedback.get('information_accuracy', 0),
                'Sources Used': feedback.get('sources_used', 0),
                'User Satisfaction': feedback.get('user_satisfaction', 0),
                'Response Time': feedback.get('response_time', 0),
                'Areas for Improvement': feedback.get('areas_for_improvement', ''),
                'Suggested Improvement': feedback.get('suggested_improvement', '')
            }
            
            # Append to DataFrame
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            
            # Save to Excel
            df.to_excel(self.feedback_file, index=False)
            
            logger.info(f"Added feedback for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False
    
    def get_agent_performance(self, agent: str) -> Dict[str, Any]:
        """
        Get performance metrics for an agent
        
        Args:
            agent: The agent name or type
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Read data
            df = pd.read_excel(self.feedback_file)
            
            # Filter by agent
            agent_df = df[df['Agent'] == agent]
            
            # Check if any data exists
            if agent_df.empty:
                return {
                    'agent': agent,
                    'average_quality': 0,
                    'average_accuracy': 0,
                    'average_satisfaction': 0,
                    'average_response_time': 0,
                    'total_responses': 0
                }
            
            # Calculate metrics
            metrics = {
                'agent': agent,
                'average_quality': agent_df['Response Quality'].mean(),
                'average_accuracy': agent_df['Information Accuracy'].mean(),
                'average_satisfaction': agent_df['User Satisfaction'].mean(),
                'average_response_time': agent_df['Response Time'].mean(),
                'total_responses': len(agent_df),
                'top_areas_for_improvement': agent_df['Areas for Improvement'].value_counts().head(3).to_dict()
            }
            
            logger.info(f"Retrieved performance metrics for agent: {agent}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting agent performance: {e}")
            return {
                'agent': agent,
                'error': str(e)
            }
    
    def get_all_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for all agents
        
        Returns:
            Dictionary of performance metrics by agent
        """
        try:
            # Read data
            df = pd.read_excel(self.feedback_file)
            
            # Get unique agents
            agents = df['Agent'].unique()
            
            # Calculate metrics for each agent
            metrics = {}
            for agent in agents:
                metrics[agent] = self.get_agent_performance(agent)
            
            # Calculate overall metrics
            metrics['overall'] = {
                'agent': 'Overall',
                'average_quality': df['Response Quality'].mean(),
                'average_accuracy': df['Information Accuracy'].mean(),
                'average_satisfaction': df['User Satisfaction'].mean(),
                'average_response_time': df['Response Time'].mean(),
                'total_responses': len(df)
            }
            
            logger.info("Retrieved performance metrics for all agents")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting all performance metrics: {e}")
            return {
                'error': str(e)
            }
    
    def get_performance_trend(self, agent: str, metric: str) -> List[Dict[str, Any]]:
        """
        Get performance trend for an agent
        
        Args:
            agent: The agent name or type
            metric: The metric to track
            
        Returns:
            List of data points for the trend
        """
        try:
            # Read data
            df = pd.read_excel(self.feedback_file)
            
            # Filter by agent
            agent_df = df[df['Agent'] == agent].copy()
            
            # Check if any data exists
            if agent_df.empty:
                return []
            
            # Convert timestamp to datetime
            agent_df['Timestamp'] = pd.to_datetime(agent_df['Timestamp'])
            
            # Sort by timestamp
            agent_df.sort_values('Timestamp', inplace=True)
            
            # Create trend data
            trend = []
            for _, row in agent_df.iterrows():
                trend.append({
                    'timestamp': row['Timestamp'].isoformat(),
                    'value': row.get(metric, 0)
                })
            
            logger.info(f"Retrieved performance trend for agent: {agent}, metric: {metric}")
            return trend
            
        except Exception as e:
            logger.error(f"Error getting performance trend: {e}")
            return []