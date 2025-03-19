"""
Memory system for CrewAI agents to track common references and technical facts
"""

from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AgentMemory:
    def __init__(self, memory_file="G:/APPS/Technical Assistant/agent_memory.json"):
        """
        Initialize agent memory system
        
        Args:
            memory_file: Path to the memory storage file
        """
        self.memory_file = memory_file
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """
        Load agent memory from file
        
        Returns:
            Dictionary containing memory data
        """
        if not os.path.exists(self.memory_file):
            # Create initial memory structure
            memory = {
                "common_products": {},
                "recent_questions": [],
                "technical_facts": {},
                "last_updated": datetime.now().isoformat()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # Save initial memory
            try:
                with open(self.memory_file, 'w') as f:
                    json.dump(memory, f, indent=2)
            except Exception as e:
                logger.error(f"Error creating initial memory file: {e}")
            
            return memory
        
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading agent memory: {e}")
            return {
                "common_products": {},
                "recent_questions": [],
                "technical_facts": {},
                "last_updated": datetime.now().isoformat()
            }
    
    def _save_memory(self):
        """Save agent memory to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # Update last_updated timestamp
            self.memory["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
                
            logger.info("Agent memory saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving agent memory: {e}")
    
    def add_product_reference(self, product_name: str, source_document: str):
        """
        Add or update a product reference in memory
        
        Args:
            product_name: Name or code of the product
            source_document: Document where the product was referenced
        """
        # Normalize product name to uppercase
        product_name = product_name.upper()
        
        if product_name not in self.memory["common_products"]:
            # Create new product entry
            self.memory["common_products"][product_name] = {
                "sources": [source_document],
                "mention_count": 1,
                "last_mentioned": datetime.now().isoformat()
            }
        else:
            # Update existing entry
            product = self.memory["common_products"][product_name]
            product["mention_count"] += 1
            product["last_mentioned"] = datetime.now().isoformat()
            
            # Add source document if not already included
            if source_document not in product["sources"]:
                product["sources"].append(source_document)
        
        # Save changes
        self._save_memory()
        
        logger.info(f"Added/updated product reference: {product_name}")
    
    def add_technical_fact(self, key: str, fact: str, source: str):
        """
        Add a technical fact to memory
        
        Args:
            key: Unique identifier for the fact
            fact: The technical fact to store
            source: Source of the fact
        """
        self.memory["technical_facts"][key] = {
            "fact": fact,
            "source": source,
            "added": datetime.now().isoformat()
        }
        
        # Save changes
        self._save_memory()
        
        logger.info(f"Added technical fact: {key}")
    
    def add_question(self, question: str):
        """
        Add a question to recent questions
        
        Args:
            question: The question to add
        """
        self.memory["recent_questions"].insert(0, {
            "question": question,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the 20 most recent questions
        self.memory["recent_questions"] = self.memory["recent_questions"][:20]
        
        # Save changes
        self._save_memory()
        
        logger.info(f"Added question to memory: {question[:50]}...")
    
    def get_product_info(self, product_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a product from memory
        
        Args:
            product_name: Name or code of the product
            
        Returns:
            Product information or None if not found
        """
        # Normalize product name to uppercase
        product_name = product_name.upper()
        
        return self.memory["common_products"].get(product_name)
    
    def get_recent_questions(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent questions from memory
        
        Args:
            count: Number of questions to retrieve
            
        Returns:
            List of recent questions
        """
        return self.memory["recent_questions"][:count]
    
    def get_technical_fact(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a technical fact from memory
        
        Args:
            key: Unique identifier for the fact
            
        Returns:
            Technical fact or None if not found
        """
        return self.memory["technical_facts"].get(key)
    
    def get_all_products(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all products in memory
        
        Returns:
            Dictionary of all products
        """
        return self.memory["common_products"]
    
    def get_all_technical_facts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all technical facts in memory
        
        Returns:
            Dictionary of all technical facts
        """
        return self.memory["technical_facts"]