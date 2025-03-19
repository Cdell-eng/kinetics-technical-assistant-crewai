import os
import json
import anthropic
from typing import List, Dict, Optional, Any, Tuple, Literal
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Body, Form
from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator, Field
import uvicorn
from pathlib import Path
import logging
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
)
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError
import pythoncom
import win32com.client
from datetime import datetime
import PyPDF2
import docx
import tempfile
import comtypes.client
from starlette.background import BackgroundTask

from fastapi.middleware.cors import CORSMiddleware
import io
import uuid
import tiktoken
import aiohttp
import azure.functions as func
from azure.search.documents.indexes import SearchIndexerClient
from config import Config
import traceback
import base64
from urllib.parse import urlparse, unquote, quote_plus
from openai import OpenAI
from enum import Enum
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest as QdrantSearchRequest
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import models, Filter, Match, MatchValue
from sentence_transformers import SentenceTransformer
import gradio as gr
import asyncio
import requests
from bs4 import BeautifulSoup
import re
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
from contextlib import asynccontextmanager
import threading
import httpx
from tempfile import SpooledTemporaryFile
import pandas as pd
import openpyxl
import time
import random
import socket

# Import CrewAI components
from Crew import (
    create_kinetics_crew,
    create_kinetics_tasks,
    create_technical_specification_crew,
    create_installation_crew,
    select_appropriate_crew,
    get_crew_response,
    ResponseCache,
    AgentMemory,
    AgentFeedback,
    with_error_handling,
    crew_fallback
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Initialize embedding model
embedding_model = SentenceTransformer('all-minilm-L6-v2')

# Initialize Azure Storage client using Config
blob_service_client = BlobServiceClient.from_connection_string(Config.AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(Config.CONTAINER_NAME)

# Set the OpenAI API key from config
os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY

# Initialize OpenAI client
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

# Initialize CrewAI components
response_cache = ResponseCache()
agent_memory = AgentMemory()
agent_feedback = AgentFeedback()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code (runs before the app starts)
    threading.Thread(target=launch_gradio, daemon=True).start()
    
    # Initialize CrewAI if enabled
    if Config.ENABLE_CREWAI:
        logger.info("Initializing CrewAI workflow")
        # Pre-load the necessary models to reduce cold start time
        try:
            test_agents = create_kinetics_crew(
                anthropic_api_key=Config.ANTHROPIC_API_KEY,
                openai_api_key=Config.OPENAI_API_KEY,
                grok_api_key=Config.GROK_API_KEY,
                query="test initialization"
            )
            logger.info("CrewAI initialized successfully")
        except Exception as e:
            logger.error(f"CrewAI initialization error: {str(e)}")
    
    yield  # This line is where the app runs
    
    # Shutdown code (runs when the app is shutting down)
    # Add any cleanup code here if needed
    pass

app = FastAPI(lifespan=lifespan)

# Add CORS middleware with more permissive settings for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null", "http://localhost:8000", "http://127.0.0.1:8000", "file://", "http://[::1]:8000"],
    allow_credentials=True,
    allow_methods=["*", "GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*", "Content-Type", "Accept", "Authorization", "Origin", "X-Requested-With"],
    max_age=3600,
    expose_headers=["*"]
)

# Initialize templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Anthropic client
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=Config.ANTHROPIC_API_KEY,
)

def get_qdrant_client():
    """Get Qdrant cloud client instance."""
    try:
        # Clean and validate URL
        url = Config.QDRANT_URL.strip()
        if '/dashboard' in url:  # Remove dashboard from URL if present
            url = url.split('/dashboard')[0]
        if url.endswith('/'):  # Remove trailing slash
            url = url[:-1]
            
        logger.info(f"Connecting to Qdrant at: {url}")
        
        client = QdrantClient(
            url=url,
            api_key=Config.QDRANT_API_KEY.strip(),
            timeout=30,  # Increased timeout
            prefer_grpc=False,  # Force HTTP
            https=True  # Force HTTPS
        )
        
        # Test connection
        collections = client.get_collections()
        logger.info(f"Connected to Qdrant. Available collections: {collections}")
        
        # Create collection if needed
        try:
            client.get_collection("knc_documents")
            logger.info("Found existing knc_documents collection")
        except Exception:
            logger.info("Creating knc_documents collection")
            client.recreate_collection(
                collection_name="knc_documents",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        
        return client
            
    except Exception as e:
        logger.error(f"Qdrant cloud connection failed: {str(e)}")
        logger.error(f"Full error: {traceback.format_exc()}")
        return None

class NetworkPathRequest(BaseModel):
    network_path: str

class DatabaseChoice(Enum):
    # AZURE = "azure"  # Commented out Azure
    QDRANT = "qdrant"

class SearchRequest(BaseModel):
    query: str
    top: int = 5
    ai_model: str = "claude"
    database: DatabaseChoice = DatabaseChoice.QDRANT

class QuestionRequest(BaseModel):
    question: str
    search_results: List[Dict]
    include_sources: bool = True
    generate_document: bool = False
    document_type: Optional[str] = None
    ai_model: str = "claude"  # New field to select AI model

class DirectoryUploadRequest(BaseModel):
    directory_path: str
    recursive: bool = True

def get_search_client():
    """Get Azure Search client."""
    try:
        # Get the Azure Search key from config
        search_key = Config.AZURE_SEARCH_KEY
        
        # Validate that we have a search key
        if not search_key:
            logger.error("AZURE_SEARCH_KEY is not set in config")
            raise ValueError("AZURE_SEARCH_KEY is not set in config")
            
        # Get the search service name from config
        search_service = Config.AZURE_SEARCH_SERVICE
        if not search_service:
            logger.error("AZURE_SEARCH_SERVICE is not set in config")
            raise ValueError("AZURE_SEARCH_SERVICE is not set in config")
            
        # Get the index name from config
        index_name = Config.AZURE_SEARCH_INDEX
        if not index_name:
            logger.error("AZURE_SEARCH_INDEX is not set in config")
            raise ValueError("AZURE_SEARCH_INDEX is not set in config")

        # Log connection attempt
        logger.info(f"Connecting to Azure Search service: {search_service}")
        
        # Create the credential and client
        credential = AzureKeyCredential(search_key)
        endpoint = f"https://{search_service}.search.windows.net/"
        
        client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential
        )
        
        # Test the connection
        try:
            # Attempt a simple search to verify credentials
            test_results = client.search(search_text="test", top=1)
            list(test_results)  # Force evaluation of the iterator
            logger.info("Successfully connected to Azure Search")
        except Exception as e:
            logger.error(f"Failed to connect to Azure Search: {str(e)}")
            if "Forbidden" in str(e):
                logger.error("Access denied. Please check your Azure Search key and permissions")
            raise
            
        return client
        
    except Exception as e:
        logger.error(f"Error creating search client: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise ValueError(f"Failed to initialize Azure Search client: {str(e)}")

def get_blob_client():
    """Get or create Azure Blob Storage container client."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(Config.AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(Config.CONTAINER_NAME)
        
        
        # Create container if it doesn't exist
        try:
            container_client.create_container()
            logger.info(f"Created container: {Config.CONTAINER_NAME}")
        except ResourceExistsError:
            logger.info(f"Container {Config.CONTAINER_NAME} already exists")
        
        return container_client
    except Exception as e:
        logger.error(f"Error connecting to Azure Storage: {str(e)}")
        raise

async def search_kinetics_website(query: str) -> List[Dict[str, str]]:
    """Search the Kinetics website using Brave Search API."""
    try:
        # Extract product name from the query
        product_name = extract_product_name(query)
        if not product_name:
            logger.warning("No product name found in query, using full query")
            product_name = query  # Fall back to full query

        # Format the search query for Kinetics website
        search_query = f"{product_name} site:kineticsnoise.com"
        logger.info(f"Searching Brave with query: {search_query}")

        # Direct Brave Search API endpoint
        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": Config.BRAVE_SEARCH_API_KEY
        }

        params = {
            "q": search_query,
            "count": 5,
            "search_lang": "en"
        }

        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, 
                headers=headers, 
                params=params,
                timeout=10
            ) as response:
                if response.status != 200:
                    logger.error(f"Brave Search API request failed with status: {response.status}")
                    return []

                data = await response.json()
                print("Brave Search Response:", json.dumps(data, indent=2))

                results = []
                
                # Extract results from the correct path in the response
                if "web" in data and "results" in data["web"]:
                    for result in data["web"]["results"]:
                        result_data = {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "description": result.get("description", "")
                        }
                        results.append(result_data)
                        print(f"\nResult found:\n{json.dumps(result_data, indent=2)}")
                else:
                    logger.warning("No results found in Brave search response")
                    
                print(f"Extracted {len(results)} results from Brave search")
                return results

    except Exception as e:
        logger.error(f"Error searching Kinetics website: {str(e)}")
        return []

def extract_product_name(query: str) -> str:
    """Extract product name from the query using regex."""
    # Comprehensive regex pattern for Kinetics products with balanced parentheses
    pattern = r'\b(?:ICW|KSCH|KIP|ESR|ACOUSTIC CURTAINS|SOUND ABSORBERS|NOISE BARRIERS|' \
             r'NOISEBLOCK|SOUND DAMPING|VIBRATION CONTROL|LSM|LIFWRK|RIM|FLM|FIC|' \
             r'ULTRA QUIET SR|SR FLOORBOARD|SOUNDMATT|ISOLAYMENT|ICC|KSCH|MUTA|GOTHAM|' \
             r'SPARTA|WAVE HANGER|ISOMAX|ISOGRID|AF|WALLMAT|PC-10|UNIBRACE|IPRB|PSB-S|' \
             r'NAFP-10|ISOBRACKER|SOUND DAMP2|KSM|KSR 3\.0|KSCR|RQ|SEISMIC V-LOOPS|' \
             r'QUAKESTRUT|KHRC|KSCC|V\.LOCK|TITAN|FMS|FLSS|FLS|FHS|PS|KSPS|CRFS|' \
             r'HS-5|HS-2|HS-1|KCAB|KCCAB|KUAB|KSMS|KSMG|KSMF|KSBC|KAABS|KAABC|KMTB|' \
             r'TG|KSWC|KHRC|QUAKESTRUT|KSSB|QUIETTILE|RIM|WAVE HANGER|' \
             r'(?:ICW|KSCH|KIP|ESR)(?:\s+(?:hangers?|isolators?|mounts?|pads?))?)\b'

    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        product_name = match.group(0)
        print("Product name extracted:", product_name)  # Debug print
        return product_name
    
    print("No product name found in query:", query)  # Debug print
    return ""

def determine_best_link(results: List[Dict[str, str]], query: str) -> Dict[str, str]:
    """Determine the best link to follow based on the full query."""
    # Example logic to determine the best link (customize as needed)
    for result in results:
        if query.lower() in result['title'].lower() or query.lower() in result['description'].lower():
            return result
    return {}

async def enhance_search_result(result: Dict, query: str) -> Dict:
    """Enhance web search result with additional content from the page, focusing on query relevance."""
    try:
        enhanced = result.copy()
        url = result.get("url", "")
        query_keywords = set(query.lower().split())
        
        print(f"Enhancing result: {url}")
        print(f"  Query keywords: {query_keywords}")
        
        if not url:
            print("  No URL provided, skipping enhancement")
            return enhanced
            
        # Handle direct binary files
        if any(url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
            print(f"  Direct binary file link detected, adding as PDF link")
            enhanced["page_content"] = enhanced.get("description", "") or "Binary file content"
            enhanced["pdf_links"] = [{
                "title": enhanced.get("title", "Document"),
                "url": url,
                "context": enhanced.get("description", ""),
                "relevance_score": 0.9  # High relevance for direct links
            }]
            enhanced["relevance_score"] = 0.9
            return enhanced
            
        # Handle URLs with file parameters
        if "file=" in url and any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx']):
            print(f"  Binary file parameter detected, adding as PDF link")
            enhanced["page_content"] = enhanced.get("description", "") or "Binary file content referenced"
            enhanced["pdf_links"] = [{
                "title": enhanced.get("title", "Document"),
                "url": url,
                "context": enhanced.get("description", ""),
                "relevance_score": 0.9  # High relevance
            }]
            enhanced["relevance_score"] = 0.9
            return enhanced
            
        # Fetch page content
        async with aiohttp.ClientSession() as session:
            try:
                print(f"  Fetching content from URL...")
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        # Handle encoding properly
                        html_bytes = await response.read()
                        try:
                            html = html_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                html = html_bytes.decode('latin-1')
                            except UnicodeDecodeError:
                                print(f"  Unable to decode page content, using description only")
                                enhanced["page_content"] = enhanced.get("description", "")
                                enhanced["relevance_score"] = 0.5
                                return enhanced
                        
                        print(f"  Received HTML content, length: {len(html)}")
                        
                        # Extract content with BeautifulSoup
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract all text paragraphs with their relevance score
                        paragraphs = []
                        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'div']):
                            text = tag.get_text(strip=True)
                            if len(text) > 30:  # Skip short content
                                # Calculate relevance score based on keyword matches
                                tag_words = set(text.lower().split())
                                keyword_matches = len(query_keywords.intersection(tag_words))
                                relevance = min(1.0, keyword_matches / len(query_keywords) if query_keywords else 0)
                                
                                # Boost score for headings
                                if tag.name in ['h1', 'h2', 'h3', 'h4']:
                                    relevance += 0.2
                                    
                                # Add paragraph to list with score
                                paragraphs.append({
                                    'text': text,
                                    'relevance': relevance
                                })
                        
                        # Sort paragraphs by relevance
                        paragraphs.sort(key=lambda x: x['relevance'], reverse=True)
                        
                        # Combine top paragraphs (up to 5)
                        main_content = "\n\n".join([p['text'] for p in paragraphs[:5]])
                        
                        # If no relevant paragraphs, use description
                        if not main_content and enhanced.get("description"):
                            main_content = enhanced.get("description")
                        
                        # Calculate overall page relevance
                        page_relevance = max([p['relevance'] for p in paragraphs[:5]]) if paragraphs else 0.5
                        
                        enhanced["page_content"] = main_content
                        enhanced["relevance_score"] = page_relevance
                        print(f"  Extracted {len(main_content)} chars with relevance score {page_relevance:.2f}")
                        
                        # Find PDF links and score them
                        pdf_links = []
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            if '.pdf' in href.lower():
                                link_text = link.get_text(strip=True)
                                context = link.parent.get_text(strip=True) if link.parent else ""
                                
                                # Score PDF relevance
                                pdf_text = f"{link_text} {context}".lower()
                                pdf_words = set(pdf_text.split())
                                keyword_matches = len(query_keywords.intersection(pdf_words))
                                pdf_relevance = min(1.0, keyword_matches / len(query_keywords) if query_keywords else 0)
                                
                                # Boost score for obvious product documentation
                                if any(term in pdf_text for term in ['spec', 'manual', 'guide', 'instruction', 'data']):
                                    pdf_relevance += 0.3
                                
                                # Add PDF with relevance score
                                pdf_info = {
                                    "title": link_text or "PDF Document",
                                    "url": href if href.startswith('http') else f"{url.rstrip('/')}/{href.lstrip('/')}",
                                    "context": context,
                                    "relevance_score": min(1.0, pdf_relevance)  # Cap at 1.0
                                }
                                pdf_links.append(pdf_info)
                        
                        # Sort PDFs by relevance
                        pdf_links.sort(key=lambda x: x['relevance_score'], reverse=True)
                        enhanced["pdf_links"] = pdf_links[:3]  # Keep top 3
                        
                        if pdf_links:
                            print(f"  Found {len(pdf_links)} PDF links, top relevance: {pdf_links[0]['relevance_score']:.2f}")
                        
                    else:
                        print(f"  Failed to fetch page: HTTP {response.status}")
                        enhanced["page_content"] = enhanced.get("description", "")
                        enhanced["relevance_score"] = 0.3
                        
            except Exception as e:
                print(f"  Error enhancing search result: {str(e)}")
                enhanced["page_content"] = enhanced.get("description", "")
                enhanced["relevance_score"] = 0.3
                logger.error(f"Error enhancing search result {url}: {str(e)}")
                
        # Add PDF processing
        if "pdf_links" in enhanced and enhanced["pdf_links"]:
            for pdf in enhanced["pdf_links"]:
                try:
                    # Download PDF content
                    pdf_content = await download_pdf(pdf["url"])
                    if pdf_content:
                        # Upload to Qdrant web_docs collection
                        upload_result = await upload_web_pdf_to_qdrant(pdf_content, pdf)
                        process_pdfs(pdf_content, pdf)
                        pdf["qdrant_status"] = upload_result["status"]
                        if upload_result["status"] == "success":
                            logger.info(f"Successfully uploaded web PDF: {pdf['title']}")
                except Exception as e:
                    logger.error(f"Error processing web PDF {pdf['url']}: {str(e)}")
                    pdf["qdrant_status"] = "error"
                
        return enhanced
        
    except Exception as e:
        print(f"Error in enhance_search_result: {str(e)}")
        logger.error(f"Error in enhance_search_result: {str(e)}")
        result["page_content"] = result.get("description", "")
        result["relevance_score"] = 0.1
        return result

# Function to determine if CrewAI should be used for a query
def determine_if_crew_needed(query: str) -> bool:
    """Determine if the query would benefit from the CrewAI workflow"""
    # Complex technical questions benefit from CrewAI
    complex_indicators = [
        "compare", "difference between", "installation", "technical specification",
        "requirements", "test", "certificate", "procedure", "step by step", 
        "design", "calculate", "recommendation", "compatibility", "how to",
        "what is the best", "which product", "noise reduction", "vibration", 
        "acoustics", "properties", "performance"
    ]
    
    # Check for complexity indicators
    for indicator in complex_indicators:
        if indicator.lower() in query.lower():
            return True
    
    # Check for product-specific questions that might need multiple sources
    product_mentioned = re.search(r'\b(?:ICW|KSCH|KIP|ESR|ACOUSTIC CURTAINS|SOUND ABSORBERS|NOISE BARRIERS|' \
             r'NOISEBLOCK|SOUND DAMPING|VIBRATION CONTROL|LSM|LIFWRK|RIM|FLM|FIC)\b', query, re.IGNORECASE)
    
    if product_mentioned and len(query.split()) > 8:  # More than 8 words with a product mention
        return True
        
    # Default to regular workflow for simple queries
    return False

# Function to randomly select workflow method for A/B testing
def should_use_crewai_for_ab_testing(query: str) -> bool:
    """Randomly select workflow method for A/B testing"""
    if not Config.ENABLE_AB_TESTING:
        # Use the default behavior based on query complexity
        return Config.ENABLE_CREWAI and determine_if_crew_needed(query)
    
    # For A/B testing, randomly select with 50% probability
    use_crew = random.random() < 0.5
    
    # Log the selection for analysis
    logger.info(f"A/B Testing: {'CrewAI' if use_crew else 'Standard'} workflow selected for query: {query[:50]}...")
    
    return use_crew

async def process_and_upload_pdfs_to_qdrant(pdf_links):
    """Process PDFs and upload them to Qdrant."""
    try:
        for pdf in pdf_links:
            try:
                pdf_url = pdf["url"]
                pdf_title = pdf["title"]
                logger.info(f"Processing PDF: {pdf_title} ({pdf_url})")
                
                # Check if PDF is already in Qdrant to avoid duplicates
                if await is_pdf_in_qdrant(pdf_url):
                    logger.info(f"PDF already in Qdrant: {pdf_title}")
                    continue
                
                # Download the PDF
                pdf_content = await download_pdf(pdf_url)
                if not pdf_content:
                    logger.warning(f"Failed to download PDF: {pdf_url}")
                    continue
                
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(pdf_content)
                if not pdf_text or len(pdf_text) < 50:  # Ensure we have meaningful content
                    logger.warning(f"No meaningful text extracted from PDF: {pdf_url}")
                    continue
                
                # Generate embeddings and upload to Qdrant
                await upload_to_qdrant(pdf_text, {
                    "metadata_storage_name": pdf_title,
                    "metadata_storage_path": pdf_url,
                    "source": "kinetics_website",
                    "content": pdf_text[:5000],  # Limit content size for storage
                    "url": pdf_url,
                    "title": pdf_title,
                    "context": pdf.get("context", "")
                })
                
                logger.info(f"Successfully processed and uploaded PDF: {pdf_title}")
                
            except Exception as e:
                logger.error(f"Error processing PDF {pdf.get('url', 'unknown')}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in PDF batch processing: {str(e)}")

async def get_ai_response(query: str, context: str, model: str = "claude", knc_tools_response: Optional[Dict] = None) -> str:
    """Get answer from AI model using provided context."""
    try:
        # Search the website once
        web_results = await search_kinetics_website(query)
        
        # Construct prompt (shared between models)
        prompt = construct_ai_prompt(query, context, web_results, knc_tools_response)
        
        # Dispatch to appropriate model
        if model.startswith("claude"):
            return await get_claude_response(prompt, model)
        elif model.startswith("gpt"):
            return await get_openai_response(prompt, model)
        elif model.startswith("grok"):
            return await get_grok_response(prompt)
        else:
            return await get_claude_response(prompt)
            
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        return f"Error: Unable to generate response due to: {str(e)}"

def truncate_context(context: str, max_tokens: int) -> str:
    """Truncate context to stay within token limit."""
    try:
        # Initialize tokenizer
        enc = tiktoken.get_encoding("cl100k_base")  # Claude's encoding
        
        # Get tokens
        tokens = enc.encode(context)
        
        if len(tokens) <= max_tokens:
            return context
            
        # Truncate tokens and decode
        truncated_tokens = tokens[:max_tokens]
        return enc.decode(truncated_tokens)
        
    except Exception as e:
        logger.error(f"Error truncating context: {str(e)}")
        # Fallback to simple character-based truncation
        return context[:max_tokens * 4]  # Rough estimate of 4 chars per token

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

async def search_documents(request: SearchRequest) -> Dict:
    """Search documents in Qdrant."""
    try:
        # if request.database == DatabaseChoice.AZURE:
        #     return await search_azure(request)
        # else:
        return await search_qdrant(request.query)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

async def search_qdrant(query: str) -> List[Dict]:
    """Search documents in Qdrant with improved content extraction."""
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        
      # Convert query to vector
        query_vector = embedding_model.encode(query, convert_to_tensor=False).tolist()
        
        # Search in Qdrant with increased limit and score threshold
        search_results = client.search(
            collection_name="knc_documents",
            query_vector=query_vector,
            limit=8,  # Increased from 5 to get more potential matches
            score_threshold=0.5  # Only include reasonably relevant results
        )
        
        # Format results with full metadata and complete content
        results = []
        for result in search_results:
            if hasattr(result, 'payload'):
                # Extract full content without truncation
                content = result.payload.get("content", "")
                
                # Log content length for debugging
                doc_name = result.payload.get("metadata_storage_name", "Unknown")
                logger.info(f"Retrieved document: {doc_name} with {len(content)} characters")
                
                results.append({
                    "content": content,  # Include full content
                    "metadata_storage_name": doc_name,
                    "metadata_storage_path": result.payload.get("metadata_storage_path", ""),
                    "score": result.score  # Include relevance score
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        print(f"Qdrant search found {len(results)} documents")
        return results
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

async def extract_text_and_images_from_file(file: UploadFile) -> Dict:
    """Extract text content and images from uploaded files."""
    content = await file.read()
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    try:
        result = {
            "text": "",
            "images": [],
            "figures": [],
            "captions": []
        }

        if file_extension == '.pdf':
            # PDF Processing
            pdf_file = io.BytesIO(content)
            
            # Extract text using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Reset file pointer for pdfplumber
            pdf_file.seek(0)
            
            # Extract images using PyMuPDF
            try:
                result["images"] = extract_images_from_pdf(content)
            except Exception as pdf_err:
                logger.warning(f"Error in PyMuPDF processing: {str(pdf_err)}")
                # Continue with just the text content
            
            result["text"] = text

        elif file_extension == '.docx':
            # Word Document Processing
            doc_file = io.BytesIO(content)
            doc = docx.Document(doc_file)
            
            # Extract text and track images
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
                
                # Check for inline images
                for run in paragraph.runs:
                    if run._element.findall('.//w:drawing') or run._element.findall('.//w:pict'):
                        result["images"].append({
                            "type": "inline",
                            "paragraph": len(text)
                        })
            
            # Extract images from document parts
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    image_data = rel._target.blob
                    result["images"].append({
                        "data": base64.b64encode(image_data).decode(),
                        "type": "embedded"
                    })
            
            result["text"] = "\n".join(text)

        elif file_extension == '.dwg':
            # AutoCAD Processing
            text_content = await extract_autocad_text(content)
            result["text"] = text_content
            
            # Extract entities that might be figures
            pythoncom.CoInitialize()
            with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            try:
                acad = win32com.client.Dispatch("AutoCAD.Application")
                doc = acad.Documents.Open(temp_path)
                
                for entity in doc.ModelSpace:
                    if entity.ObjectName in ["AcDbBlockReference", "AcDbPolyline", "AcDbCircle"]:
                        result["figures"].append({
                            "type": entity.ObjectName,
                            "coordinates": [entity.Coordinates.Item(i) for i in range(entity.Coordinates.Count)]
                        })
                
                doc.Close()
            finally:
                os.unlink(temp_path)
                pythoncom.CoUninitialize()

        elif file_extension in ['.sldprt', '.sldasm']:
            # SolidWorks Processing
            text_content = await extract_solidworks_text(content)
            result["text"] = text_content
            
            # Extract model views and drawings
            pythoncom.CoInitialize()
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            try:
                sw = win32com.client.Dispatch("SldWorks.Application")
                doc = sw.OpenDoc(temp_path, 1)
                
                # Get model views
                view_count = doc.GetViewCount()
                for i in range(view_count):
                    view = doc.GetView(i)
                    if view:
                        result["figures"].append({
                            "type": "model_view",
                            "view_index": i,
                            "orientation": view.Orientation
                        })
                
                doc.CloseDoc()
            finally:
                os.unlink(temp_path)
                pythoncom.CoUninitialize()

        return result
            
    except Exception as e:
        logger.error(f"Error extracting content from {file.filename}: {str(e)}")
        return {"text": "", "images": [], "figures": [], "captions": []}

async def extract_autocad_text(content: bytes) -> str:
    """Extract text from AutoCAD files."""
    try:
        # Initialize COM
        pythoncom.CoInitialize()
        
        # Save content to temp file
        with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        # Create AutoCAD application
        acad = win32com.client.Dispatch("AutoCAD.Application")
        doc = acad.Documents.Open(temp_path)
        
        # Extract text from all text objects
        text_content = []
        for entity in doc.ModelSpace:
            if entity.ObjectName in ["AcDbText", "AcDbMText"]:
                text_content.append(entity.TextString)
                
        doc.Close()
        os.unlink(temp_path)
        
        return "\n".join(text_content)
        
    except Exception as e:
        logger.error(f"AutoCAD extraction error: {str(e)}")
        return ""
    finally:
        pythoncom.CoUninitialize()

async def extract_solidworks_text(content: bytes) -> str:
    """Extract text from SolidWorks files."""
    try:
        # Initialize COM
        pythoncom.CoInitialize()
        
        # Save content to temp file
        with tempfile.NamedTemporaryFile(suffix='.sldprt', delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        # Create SolidWorks application
        sw = win32com.client.Dispatch("SldWorks.Application")
        doc = sw.OpenDoc(temp_path, 1)  # 1 = swDocPART
        
        # Extract custom properties
        custom_props = doc.GetCustomInfoProperties()
        text_content = []
        
        for prop in custom_props:
            value = doc.GetCustomInfoValue(prop)
            text_content.append(f"{prop}: {value}")
            
        # Extract annotations and notes
        notes = doc.GetNotes()
        if notes:
            text_content.extend(notes)
            
        doc.CloseDoc()
        os.unlink(temp_path)
        
        return "\n".join(text_content)
        
    except Exception as e:
        logger.error(f"SolidWorks extraction error: {str(e)}")
        return ""
    finally:
        pythoncom.CoUninitialize()

async def process_and_upload_file(file: UploadFile, container_client: ContainerClient) -> Dict:
    """Process file and upload to Azure Storage."""
    try:
        # Extract text content
        text_content = await extract_text_and_images_from_file(file)
        
        # Generate unique blob name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"{timestamp}_{file.filename}"
        
        # Create metadata
        metadata = {
            "original_filename": file.filename,
            "content_type": file.content_type,
            "upload_timestamp": timestamp,
            "has_extracted_text": "true" if text_content["text"] else "false"
        }
        
        # Upload original file
        blob_client = container_client.get_blob_client(blob_name)
        file.file.seek(0)
        content = await file.read()
        blob_client.upload_blob(content, metadata=metadata)
        
        # If text was extracted, upload it as a separate blob
        if text_content["text"]:
            text_blob_name = f"{blob_name}.txt"
            text_blob_client = container_client.get_blob_client(text_blob_name)
            text_blob_client.upload_blob(text_content["text"].encode('utf-8'))
        
        return {
            "filename": file.filename,
            "blob_name": blob_name,
            "size": len(content),
            "has_extracted_text": bool(text_content["text"])
        }
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")

@app.post("/upload")
async def upload_document(
    files: List[UploadFile] = File(...), 
    database: str = Form(...)
):
    """Upload documents to selected database(s)."""
    try:
        results = {
            "status": "success",
            "uploaded": [],
            "skipped": [],
            "failed": []
        }
        
        for file in files:
            try:
                # Reset file position before processing
                await file.seek(0)
                
                if database in ["both", "qdrant"]:
                    # Process for Qdrant
                    qdrant_result = await upload_to_qdrant(file)
                
                if qdrant_result.get("status") == "success":
                        results["uploaded"].append({
                            "filename": file.filename,
                            "database": "qdrant",
                            "document_id": qdrant_result.get("document_id")
                        })
                elif qdrant_result.get("status") == "skipped":
                        results["skipped"].append({
                            "filename": file.filename,
                            "reason": qdrant_result.get("message")
                        })
                else:
                        results["failed"].append({
                            "filename": file.filename,
                            "error": qdrant_result.get("message")
                        })
            
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                results["failed"].append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# Define response models
class Source(BaseModel):
    file_name: str
    file_path: str
    context: str
    storage_path: str

class AnswerResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = None

async def generate_technical_document(question: str, context: str, document_type: str) -> str:
    """Generate technical documentation based on examples and context."""
    try:
        Gprompt = f"""You are a technical documentation specialist for Kinetics Noise Control. Generate a {document_type} based on the question and similar examples from the context. Follow the standard formatting and structure shown in the examples. Only use information from the provided context. in your response, when listing information,create a separate line for each item.

Context with examples:
{context}

Request: {question}

Please generate a detailed {document_type} following the same format and standards as shown in the examples. Include all necessary technical specifications, measurements, and requirements."""

        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,  # Increased for longer documents
            temperature=0.2,  # Slight creativity while maintaining accuracy
            messages=[
                {"role": "user", "content": Gprompt}
            ]
        )
        
        return message.content[0].text
        
    except Exception as e:
        logger.error(f"Error generating technical document: {str(e)}")
        return f"Error generating document: {str(e)}"



async def get_answer_based_on_model(question: str, context: str, ai_model: str) -> str:
    """Get answer from the selected AI model based on question and context."""
    if ai_model == "openai":
        return await get_ai_response(question, context, ai_model)
    else:
        return await get_ai_response(question, context)

# Define the model options
AVAILABLE_MODELS = [
    "claude-3-5-sonnet-20240620",  # Default Claude model
    "claude-3-5-sonnet-20241022", # Claude 3.5 Sonnet
    "claude-3-opus-20240229",     # Claude 3 Opus
    "gpt-4o-mini",         # OpenAI GPT-4 Turbo
    "gpt-4o",              # OpenAI GPT-4o
    "grok-2-latest"        # xAI Grok-2
]

DEFAULT_MODEL = "grok-2-latest"

# Define custom theme for Kinetics branding
theme = gr.themes.Default().set(
    body_background_fill="#f3f4f6",  # Light gray background
    block_background_fill="#ffffff",  # White blocks
    button_primary_background_fill="#004B87",  # Kinetics blue
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
)

# Update the UI layout to include a model selector
with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                value=[],
                type="messages",
                label="Chatbot",
                height=700,
                show_copy_button=True,
                render_markdown=True,
            )
            
            with gr.Row():
                txt = gr.Textbox(
                    placeholder="Ask me anything about Kinetics products and technical documentation...",
                    scale=8,
                    container=False,
                    show_label=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)
            
                model_selector = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_MODEL,
                label="AI Model"
                )
        
        with gr.Column(scale=3):
            gr.Markdown("### Reference Documents")
            file_list = gr.Textbox(
                label="",
                value="No documents loaded",
                lines=10,
                max_lines=15,
                interactive=False,
                show_copy_button=True
            )
    
    # Update event handlers to include model selection
    async def on_send_with_model(message, history, model):
        if message:
            response, doc_text = await respond_to_message(message, history, model)
            return "", history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ], doc_text
        return "", history, ""
    
    # Connect the event handlers with model selector
    txt.submit(
        fn=on_send_with_model,
        inputs=[txt, chatbot, model_selector],
        outputs=[txt, chatbot, file_list],
        api_name="submit"
    )
    
    send_btn.click(
        fn=on_send_with_model,
        inputs=[txt, chatbot, model_selector],
        outputs=[txt, chatbot, file_list],
        api_name="send"
    )
    
    clear_btn.click(
        fn=lambda: (None, [], ""),
        outputs=[txt, chatbot, file_list],
        queue=False
    )
    
# Add new function to handle Excel logging
def setup_excel_log():
    """Create or load the Excel log file."""
    try:
        log_dir = Path("G:/APPS/Technical Assistant")
        log_file = log_dir / "chat_logs.xlsx"
        
        # Create directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if not log_file.exists():
            # Create new Excel file with headers
            df = pd.DataFrame(columns=[
                'Timestamp',
                'Question',
                'AI Response',
                'Model Used',
                'Sources Used',
                'Response Time (s)',
                'Corrected Response'
            ])
            df.to_excel(log_file, index=False)
            logger.info(f"Created new chat log file: {log_file}")
        
        return log_file
    except Exception as e:
        logger.error(f"Error setting up Excel log: {str(e)}")
        return None

# Update the log_chat_to_excel function with better error handling
async def log_chat_to_excel(question: str, response: str, model: str, sources: List[Dict] = None, response_time: float = 0) -> None:
    """Log chat interaction to Excel file with space for corrections."""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            log_file = setup_excel_log()
            if log_file is None:
                logger.error("Could not setup Excel log file")
                return
            
            # Format sources into string
            sources_str = ""
            if sources:
                source_items = []
                for s in sources:
                    if isinstance(s, dict):
                        if "type" in s and ("name" in s or "title" in s):
                            source_items.append(f"- {s.get('type')}: {s.get('name', s.get('title', 'Unknown'))}")
                        elif "file_name" in s:
                            source_items.append(f"- {s.get('file_name', 'Unknown')}")
                sources_str = "\n".join(source_items)
            
            # Create new row data
            new_row = {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Question': question,
                'AI Response': response,
                'Model Used': model,
                'Sources Used': sources_str,
                'Response Time (s)': round(response_time, 2),
                'Corrected Response': ''  # Empty column for manual corrections
            }
            
            # Read existing data
            try:
                df = pd.read_excel(log_file)
                
                # Check if dataframe is empty or malformed
                if df.empty:
                    # Create a new dataframe with the correct columns
                    df = pd.DataFrame(columns=[
                        'Timestamp',
                        'Question',
                        'AI Response',
                        'Model Used',
                        'Sources Used',
                        'Response Time (s)',
                        'Corrected Response'
                    ])
                
                # Ensure all required columns exist
                for col in new_row.keys():
                    if col not in df.columns:
                        df[col] = ""
                
                # Append new row (always append, don't try to update last row)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
            except PermissionError:
                error_msg = "Cannot access chat logs - Excel file is open. Please close the file and try again."
                logger.warning(error_msg)
                message = gr.Markdown(error_msg)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error("Max retries reached - could not save to Excel file")
                    return
            except Exception as e:
                logger.error(f"Error reading Excel file: {str(e)}")
                # Create a new dataframe with just this row
                df = pd.DataFrame([new_row])
            
            # Save back to Excel
            try:
                df.to_excel(log_file, index=False)
                logger.info(f"Successfully logged chat interaction to Excel file")
                return  # Success - exit function
                
            except PermissionError:
                error_msg = "Cannot save chat logs - Excel file is open. Please close the file and try again."
                logger.warning(error_msg)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error("Max retries reached - could not save to Excel file")
                    return
                    
        except Exception as e:
            logger.error(f"Error logging chat to Excel: {str(e)}")
            return

# Modified respond_to_message function with CrewAI integration
async def respond_to_message(message, history, model=DEFAULT_MODEL):
    """Handle chat responses with CrewAI workflow integration."""
    try:
        # Initialize document list and error tracking
        storage_names = []
        excel_log_error = None
        start_time = time.time()
        
        # Extract context from history
        context = ""
        if history:
            for msg in history[-3:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                context += f"{role.capitalize()}: {content}\n"
        
        # Determine whether to use CrewAI based on configuration and query
        use_crew = should_use_crewai_for_ab_testing(message)
        logger.info(f"Using {'CrewAI' if use_crew else 'Standard'} workflow for: {message[:50]}...")
        
        if use_crew:
            # Use CrewAI workflow
            config = {
                "ANTHROPIC_API_KEY": Config.ANTHROPIC_API_KEY,
                "OPENAI_API_KEY": Config.OPENAI_API_KEY,
                "GROK_API_KEY": Config.GROK_API_KEY,
            }
            
            crew_response = await get_crew_response(message, config)
            response = crew_response["answer"]
            sources = crew_response["sources"]
            
            # Extract storage names for display
            for source in sources:
                if "file_name" in source:
                    storage_names.append(f"• {sanitize_text(source['file_name'])}")
                elif "title" in source:
                    storage_names.append(f"• {sanitize_text(source['title'])} (Web)")
        else:
            # Use the original workflow
            # Search documents in Qdrant
            search_results = await search_qdrant(message)
            
            # Format document references and create sources list
            sources = []
            base_path = "G:/Standards/Test Reports & Product Photos"
            
            for result in search_results:
                storage_name = result.get('metadata_storage_name', '')
                if storage_name:
                    storage_names.append(f"• {sanitize_text(storage_name)}")
                
                # Create source with full path
                storage_path = result.get("metadata_storage_path", "")
                if storage_path and not storage_path.startswith(base_path):
                    storage_path = os.path.join(base_path, storage_path.lstrip("/\\"))
                
                source = {
                    "file_name": storage_name,
                    "file_path": storage_path,
                    "context": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                    "storage_path": storage_path,
                    "display_path": f"{base_path}/{storage_name}"
                }
                sources.append(source)
                logger.info(f"Added document reference: {source['display_path']}")
            
            # Format context from knowledge base with better structure
            kb_context = ""
            if search_results:
                for i, result in enumerate(search_results):
                    doc_name = result.get('metadata_storage_name', 'Unknown document')
                    content = result.get('content', '')
                    
                    # Add document separator and clear section heading
                    kb_context += f"\n=== DOCUMENT {i+1}: {doc_name} ===\n"
                    kb_context += f"{content}\n"
                    kb_context += f"=== END OF DOCUMENT {i+1} ===\n\n"
            
            # Get web search results
            web_results = await search_kinetics_website(message)
            
            # Enhance top web results 
            enhanced_web_results = []
            for i, result in enumerate(web_results[:3]):
                enhanced_result = await enhance_search_result(result, message)
                enhanced_web_results.append(enhanced_result)
                
                # Add web results to sources
                if web_results:
                    sources.extend([{
                        "type": "web",
                        "title": result.get('title', 'Unknown'),
                        "url": result.get('url', ''),
                        "pdf_links": result.get('pdf_links', [])
                    } for result in web_results])
            
            # Construct the prompt with improved formatting
            prompt = construct_ai_prompt(message, kb_context, enhanced_web_results)
            
            # Get response based on selected model
            if model.startswith("claude"):
                response = await get_claude_response(prompt, model)
            elif model.startswith("gpt"):
                response = await get_openai_response(prompt, model)
            elif model.startswith("grok"):
                response = await get_grok_response(prompt)
            else:
                response = await get_claude_response(prompt)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log the interaction to Excel with error handling
        try:
            await log_chat_to_excel(
                question=message,
                response=response,
                model="CrewAI" if use_crew else model,
                sources=sources,
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"Failed to log chat to Excel: {str(e)}")
            excel_log_error = str(e)
        
        # Format document list for display
        storage_names_text = "\n".join(storage_names) if storage_names else "No documents found"
        
        # Add warning to response if Excel logging failed
        if excel_log_error and "Permission denied" in excel_log_error:
            response += "\n\nNote: Chat logging is temporarily unavailable - the log file is open. Please close 'chat_logs.xlsx' to enable logging."
        
        # Log performance metrics
        try:
            agent_feedback.add_feedback(
                query=message,
                agent="CrewAI" if use_crew else model,
                feedback={
                    "response_quality": 4,  # Default quality score
                    "information_accuracy": 4,
                    "sources_used": len(sources),
                    "user_satisfaction": 4,
                    "response_time": response_time,
                    "areas_for_improvement": "",
                    "suggested_improvement": ""
                }
            )
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
        
        logger.info(f"Response generated in {response_time:.2f}s with {len(sources)} sources")
        return response, storage_names_text
                
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "I apologize, but I encountered an error processing your request.", "Error retrieving documents"

async def get_claude_response(prompt: str, model="claude-3-sonnet-20240229") -> str:
    """Get response from Claude API with proper error handling."""
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": Config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": model,
            "max_tokens": 4000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                # Updated to match Claude API response format
                return result.get('content', [{}])[0].get('text', 'No response generated')
            else:
                logger.error(f"Claude API error: {response.status_code} - {response.text}")
                return f"Error: Unable to get response (Status: {response.status_code})"
                
    except Exception as e:
        logger.error(f"Error in Claude response: {str(e)}")
        return f"Error: {str(e)}"

async def get_openai_response(prompt: str, model="gpt-4-1106-preview") -> str:
    """Get response from OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            model=model,  # Use the selected model
            messages=[
                {"role": "system", "content": "You are a helpful technical assistant for Kinetics Noise Control."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting OpenAI response: {str(e)}")
        return f"I apologize, but I'm having trouble connecting to my knowledge service. Please try again later. (Error: {type(e).__name__})"

async def get_grok_response(prompt: str) -> str:
    """Get response from Grok API."""
    try:
        client = OpenAI(
            api_key=Config.GROK_API_KEY,
            base_url="https://api.x.ai/v1"
        )
        
        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical assistant for Kinetics Noise Control."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting Grok response: {str(e)}")
        return f"I apologize, but I'm having trouble connecting to Grok. Please try again later. (Error: {type(e).__name__})"
    
# Unified PDF processing with better caching
async def process_pdfs(pdf_links: List[Dict], max_workers: int = 3):
    """Process PDFs in parallel using efficient batching."""
    if not pdf_links:
        return
        
    # Filter out already processed PDFs to avoid redundant work
    new_links = []
    for pdf in pdf_links:
        if not await is_pdf_in_qdrant(pdf["url"]):
            new_links.append(pdf)
            
    if not new_links:
        logger.info("No new PDFs to process")
        return
        
    # Process in parallel batches
    async with asyncio.TaskGroup() as tg:
        for pdf in new_links:
            tg.create_task(process_single_pdf(pdf))

async def is_pdf_in_qdrant(pdf_url):
    """Check if a PDF is already in Qdrant by URL."""
    try:
        qdrant_client = get_qdrant_client()
        if not qdrant_client:
            return False
            
        # Search by URL
        search_results = qdrant_client.search(
            collection_name="knc_documents",
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata_storage_path",
                        match=models.MatchValue(value=pdf_url)
                    )
                ]
            ),
            limit=1
        )
        
        return len(search_results) > 0
        
    except Exception as e:
        logger.error(f"Error checking Qdrant for PDF {pdf_url}: {str(e)}")
        return False

async def process_single_pdf(pdf):
    """Process a single PDF and upload to Qdrant."""
    try:
        pdf_url = pdf["url"]
        pdf_title = pdf["title"]
        logger.info(f"Processing PDF: {pdf_title} ({pdf_url})")
        
        # Download the PDF
        pdf_content = await download_pdf(pdf_url)
        if not pdf_content:
            logger.warning(f"Failed to download PDF: {pdf_url}")
            return
            
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_content)
        if not pdf_text or len(pdf_text) < 50:  # Ensure we have meaningful content
            logger.warning(f"No meaningful text extracted from PDF: {pdf_url}")
            return
        
        # Generate embeddings and upload to Qdrant
        await upload_to_qdrant(pdf_text, {
            "metadata_storage_name": pdf_title,
            "metadata_storage_path": pdf_url,
            "source": "kinetics_website",
            "content": pdf_text[:5000],  # Limit content size for storage
            "url": pdf_url,
            "title": pdf_title,
            "context": pdf.get("context", "")
        })
        
        logger.info(f"Successfully processed and uploaded PDF: {pdf_title}")
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf.get('url', 'unknown')}: {str(e)}")

async def download_pdf(url):
    """Download PDF content from URL with improved error handling and URL normalization."""
    try:
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            # Handle relative URLs
            if url.startswith('/'):
                # Assume it's from kineticsnoise.com if it starts with /
                url = f"https://kineticsnoise.com{url}"
            else:
                logger.warning(f"Skipping invalid URL: {url}")
                return None
        
        logger.info(f"Downloading PDF from: {url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/pdf,*/*",
            "Referer": "https://kineticsnoise.com/"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, timeout=30, allow_redirects=True) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download PDF from {url}, status: {response.status}")
                        
                        # Try alternative URL formats if original fails
                        if "kineticsnoise.com" in url and not url.endswith('.pdf'):
                            # Try adding .pdf extension
                            alt_url = f"{url}.pdf"
                            logger.info(f"Trying alternative URL: {alt_url}")
                            
                            async with session.get(alt_url, headers=headers, timeout=30) as alt_response:
                                if alt_response.status == 200:
                                    logger.info(f"Successfully downloaded PDF from alternative URL: {alt_url}")
                                    return await alt_response.read()
                        
                        return None
                    
                    # Check if response is actually a PDF
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                        return await response.read()
                    else:
                        logger.warning(f"URL {url} returned non-PDF content: {content_type}")
                        # Try to download anyway in case Content-Type is incorrect
                        content = await response.read()
                        # Check if content starts with PDF signature %PDF
                        if content.startswith(b'%PDF'):
                            return content
                        else:
                            logger.error(f"Content from {url} is not a valid PDF")
                            return None
            except asyncio.TimeoutError:
                logger.error(f"Timeout downloading PDF from {url}")
                return None
    except Exception as e:
        logger.error(f"Error downloading PDF {url}: {str(e)}")
        return None

def extract_text_from_pdf(pdf_content):
    """Extract text from PDF content."""
    try:
        # Try PyMuPDF first
        text = ""
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name
                
            doc = fitz.open(tmp_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            
            os.unlink(tmp_path)
            
        except Exception as e1:
            logger.warning(f"PyMuPDF failed, trying pdfminer: {str(e1)}")
            
            # Fallback to pdfminer
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name
                
            text = extract_text(tmp_path)
            os.unlink(tmp_path)
            
        return text
                    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

async def upload_to_qdrant(file: UploadFile) -> Dict:
    """Upload document with content to Qdrant."""
    try:
        logger.info(f"Starting Qdrant upload for file: {file.filename}")
        
        # Check if file already exists in Qdrant
        collection_name = "knc_documents"
        qdrant_client = get_qdrant_client()
        
        try:
            # Search for existing file by filename
            search_results = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata_storage_name",
                            match=models.MatchValue(value=file.filename)
                        )
                    ]
                )
            )
            
            if search_results[0]:  # If any results found
                return {
                    "status": "skipped",
                    "message": f"File {file.filename} already exists in database",
                    "file": file.filename
                }
        except Exception:
            # Collection doesn't exist yet, create it later
            pass

        # Get content and embeddings
        content = await extract_text_and_images_from_file(file)
        if not content["text"] and not content["images"] and not content["figures"]:
            return {
                "status": "skipped",
                "message": f"No content could be extracted from {file.filename}",
                "file": file.filename
            }

        # Store original file content
        file.file.seek(0)
        file_content = await file.read()
        file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        
        # Generate base path for documents
        base_path = "G:/Standards/Test Reports & Product Photos"
        storage_path = os.path.dirname(file.filename)  # Get storage path or empty string
        display_path = os.path.join(base_path, storage_path.lstrip("/\\")) if storage_path else base_path

        # Generate embeddings for text
        text_embedding = embedding_model.encode(content["text"]) if content["text"] else None
        image_embedding = embedding_model.encode(content["images"]) if content["images"] else None
        # Create point with enhanced metadata and file content
        point = {
            "id": str(uuid.uuid4()),
            "vector": text_embedding.tolist() if text_embedding is not None else None,
            "payload": {
                "metadata_storage_name": file.filename,
                "metadata_storage_path": storage_path,  # Keep original storage path
                "display_path": display_path,  # Use full directory path
                "content": content["text"],
                "images": content["images"],
                "figures": content["figures"],
                "captions": content["captions"],
                "upload_time": datetime.now().isoformat(),
                "file_content": file_content_b64  # Store the original file content
            }
        }

        # Upload to Qdrant
        collection_name = "knc_documents"
        qdrant_client = get_qdrant_client()
        
        try:
            qdrant_client.get_collection(collection_name)
        except Exception:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=len(text_embedding) if text_embedding is not None else 384,
                    distance=Distance.COSINE
                )
            )
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True
        )

        return {
            "status": "success",
            "message": f"Successfully uploaded {file.filename} to Qdrant",
            "document_id": point["id"]
        }

    except Exception as e:
        error_msg = f"Qdrant upload error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": error_msg,
            "file": file.filename
        }

def construct_ai_prompt(query: str, context: str, web_results: List[Dict], knc_tools_response: Optional[Dict] = None) -> str:
    """Construct an improved AI prompt with better knowledge base emphasis."""
    
    # Create a more structured prompt that prioritizes Qdrant knowledge base
    prompt_parts = [
        f"""Question: {query}

You are an expert assistant for Kinetics Noise Control. Answer the question using the provided information from the KNOWLEDGE BASE and Kinetics website. The KNOWLEDGE BASE contains authoritative internal documents and should be prioritized when they contain relevant information.

IMPORTANT INSTRUCTION: When Kinetics internal documents from the KNOWLEDGE BASE contain information relevant to the question, PRIORITIZE this information in your answer as it's the most authoritative source. Quote specific data, figures, and test results from the knowledge base documents when available.
"""
    ]
    
    # Add context from Qdrant knowledge base with clear labeling
    if context:
        prompt_parts.append(f"""
=== KNOWLEDGE BASE (INTERNAL DOCUMENTS) ===
The following information comes from Kinetics Noise Control's internal technical documents. This is highly authoritative information and should be prioritized when relevant to the question:

{context}
""")
    else:
        prompt_parts.append("""
=== KNOWLEDGE BASE (INTERNAL DOCUMENTS) ===
No relevant internal documents were found for this query.
""")
    
    # Add web search results but mark them as supplementary
    if web_results:
        prompt_parts.append("""
=== KINETICS WEBSITE INFORMATION ===
The following information comes from the public Kinetics website. Use this to supplement the knowledge base information especially when the question is about Kinetics services and does not mention testing or technical information:
""")
        for i, result in enumerate(web_results[:3], 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("page_content", "")
            prompt_parts.append(f"Source {i}: {title} ({url})\n{content[:800]}...\n\n")
            
            # Add PDF links if available
            if "pdf_links" in result and result["pdf_links"]:
                prompt_parts.append(f"Related documentation for {title}:\n")
                for pdf in result["pdf_links"]:
                    prompt_parts.append(f"- {pdf.get('title', 'Document')}: {pdf.get('url', '')}\n")
                    if "context" in pdf:
                        prompt_parts.append(f"  Description: {pdf.get('context', '')}\n")
                prompt_parts.append("\n")
    
    # Add KNC tools response if available
    if knc_tools_response:
        prompt_parts.append("""
=== ADDITIONAL TECHNICAL INFORMATION ===
""")
        prompt_parts.append(f"{json.dumps(knc_tools_response, indent=2)}\n\n")
    
    # Final instruction with emphasis on using knowledge base
    prompt_parts.append("""
ANSWER GUIDELINES:
1. Use information from the knowledge base when it's relevant to the question.
2. Use the website information if the information from the knowledge base does not completely answer the question or is not relevant. Provide links to documents and pages referenced when using information from the web search. 
3. If documents contain contradictory information, prioritize information from the knowledge base.
4. Clearly indicate which source you used (knowledge base document or website), and reference page numbers, figures, tables, and other relevant information when possible.
5. If neither source contains the complete answer, acknowledge the limitations.
6. Combine information from knowledge base and web search when both have relevant information.
7. Suggest helpful pages or PDF links from the web search.

Your response should be comprehensive, technically accurate, and clearly reference the sources you used.
""")
    
    return "".join(prompt_parts)

def extract_images_from_pdf(pdf_content):
    """Extract images from PDF content using PyMuPDF."""
    try:
        images = []
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name
            
        doc = fitz.open(tmp_path)
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                
                # Convert image data to base64 for storage/display
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                
                images.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "data": image_b64,
                    "mime": base_image["ext"]
                })
                
        doc.close()
        os.unlink(tmp_path)
        
        return images
        
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {str(e)}")
        return []

# Create a Gradio interface with CrewAI support
def create_gradio_interface():
    """Create and configure the Gradio interface."""
    try:
        with gr.Blocks(theme=theme) as demo:
            # Initialize chat state
            state = gr.State([])
            
            gr.Markdown("# Kinetics Noise Control Technical Assistant")
            
            with gr.Tabs() as tabs:
                with gr.TabItem("Chat"):
                    with gr.Row():
                        with gr.Column(scale=7):
                            # Initialize chatbot with type='messages'
                            chatbot = gr.Chatbot(
                                value=[],
                                label="Chat History",
                                height=700,
                                show_copy_button=True,
                                render_markdown=True,
                                type='messages'  # Explicitly set type to messages
                            )
                            
                            with gr.Row():
                                # Initialize message input
                                msg = gr.Textbox(
                                    show_label=False,
                                    placeholder="Ask me anything about Kinetics products...",
                                    container=False
                                )
                                submit = gr.Button("Send", variant="primary")
                                clear = gr.Button("Clear")
                            
                            with gr.Row():
                                model_selector = gr.Dropdown(
                                    choices=AVAILABLE_MODELS,
                                    value=DEFAULT_MODEL,
                                    label="AI Model",
                                    scale=2
                                )
                                
                                crewai_enabled = gr.Checkbox(
                                    label="Enable CrewAI",
                                    value=Config.ENABLE_CREWAI,
                                    interactive=True,
                                    scale=1
                                )
                            
                        with gr.Column(scale=3):
                            gr.Markdown("### Reference Documents")
                            file_list = gr.Textbox(
                                label="",
                                value="No documents loaded",
                                lines=10,
                                max_lines=15,
                                interactive=False,
                                show_copy_button=True
                            )

                    # Event handlers
                    async def chat(message, history, model):
                        """Handle chat messages with proper message format."""
                        if message:
                            # Get response from the model
                            response, doc_text = await respond_to_message(message, history, model)
                            
                            # Return updated history in the correct format
                            return "", history + [
                                {"role": "user", "content": message},
                                {"role": "assistant", "content": response}
                            ], doc_text
                        return "", history, ""
                    
                    def clear_chat():
                        """Clear chat history."""
                        return [], [], "No documents loaded"
                    
                    # Connect event handlers
                    submit.click(
                        chat,
                        inputs=[msg, state, model_selector],
                        outputs=[msg, chatbot, file_list]
                    )
                    
                    msg.submit(
                        chat,
                        inputs=[msg, state, model_selector],
                        outputs=[msg, chatbot, file_list]
                    )
                    
                    clear.click(
                        clear_chat,
                        outputs=[chatbot, state, file_list]
                    )
                    
                    # CrewAI toggle handler
                    def toggle_crewai(enabled):
                        Config.ENABLE_CREWAI = enabled
                        return f"CrewAI {'enabled' if enabled else 'disabled'}"
                    
                    crewai_status = gr.Markdown("CrewAI workflow status: enabled" if Config.ENABLE_CREWAI else "CrewAI workflow status: disabled")
                    crewai_enabled.change(toggle_crewai, inputs=[crewai_enabled], outputs=[crewai_status])
                    
                # Upload tab
                with gr.TabItem("Upload Documents"):
                    with gr.Column():
                        gr.Markdown("### Upload Documents to Knowledge Base")
                        gr.Markdown("Supported file types: PDF, DOCX, DOC, TXT, MD, XLSX, XLS")
                        
                        folder_upload = gr.File(
                            label="Upload Folder",
                            file_count="directory",
                            type="filepath"
                        )
                        
                        file_upload = gr.File(
                            label="Upload Individual Files",
                            file_count="multiple",
                            type="filepath"
                        )
                        
                        database_choice = gr.Radio(
                            choices=["qdrant", "both", "azure"],
                            value="qdrant",
                            label="Upload to Database"
                        )
                        
                        upload_button = gr.Button("Upload Files", variant="primary")
                        
                        upload_status = gr.Markdown("Upload files to add them to the knowledge base.")
                        
                        async def handle_file_upload(files: List[str], database_choice: str) -> str:
                            """Handle file uploads to the selected database."""
                            try:
                                results = {
                                    "success": [],
                                    "failed": []
                                }
                                
                                for file_path in files:
                                    try:
                                        # Create UploadFile object from file path
                                        with open(file_path, 'rb') as f:
                                            file_content = f.read()
                                            filename = os.path.basename(file_path)
                                            file = UploadFile(filename=filename, file=SpooledTemporaryFile())
                                            await file.write(file_content)
                                            await file.seek(0)
                                            
                                            # Upload to selected database
                                            if database_choice in ["qdrant", "both"]:
                                                result = await upload_to_qdrant(file)
                                                if result.get("status") == "success":
                                                    results["success"].append(f"{filename} (Qdrant)")
                                                else:
                                                    results["failed"].append(f"{filename}: {result.get('message')}")
                                            
                                            await file.close()
                                            
                                    except Exception as e:
                                        results["failed"].append(f"{os.path.basename(file_path)}: {str(e)}")
                                
                                # Format results message
                                message = []
                                if results["success"]:
                                    message.append("Successfully uploaded:\n" + "\n".join(results["success"]))
                                if results["failed"]:
                                    message.append("\nFailed uploads:\n" + "\n".join(results["failed"]))
                                    
                                return "\n\n".join(message) if message else "No files were processed."
                                
                            except Exception as e:
                                logger.error(f"Error in handle_file_upload: {str(e)}")
                                return f"Error processing uploads: {str(e)}"
                        
                        def process_uploads(folder_files, individual_files, choice):
                            """Synchronous wrapper for file upload handling"""
                            all_files = []
                            if folder_files:
                                all_files.extend(folder_files)
                            if individual_files:
                                all_files.extend(individual_files)
                            
                            if not all_files:
                                return "No files selected for upload."
                            
                            try:
                                # Create a new event loop for async operations
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                
                                # Run the async function and get results
                                results = loop.run_until_complete(handle_file_upload(all_files, choice))
                                loop.close()
                                
                                return results
                                
                            except Exception as e:
                                logger.error(f"Error processing uploads: {str(e)}")
                                return f"Error processing uploads: {str(e)}"
                        
                        # Connect the upload button to the synchronous wrapper function
                        upload_button.click(
                            fn=process_uploads,
                            inputs=[folder_upload, file_upload, database_choice],
                            outputs=upload_status
                        )
                
                # Analytics tab
                with gr.TabItem("Analytics"):
                    with gr.Column():
                        gr.Markdown("### CrewAI Performance Analytics")
                        
                        refresh_btn = gr.Button("Refresh Analytics")
                        
                        with gr.Row():
                            with gr.Column():
                                performance_chart = gr.Plot(label="Response Times")
                            
                            with gr.Column():
                                source_chart = gr.Plot(label="Sources Used")
                        
                        with gr.Row():
                            crew_stats = gr.DataFrame(label="AI Model Statistics")
                        
                        def refresh_analytics():
                            try:
                                # Read feedback data
                                feedback_file = "G:/APPS/Technical Assistant/agent_feedback.xlsx"
                                if not os.path.exists(feedback_file):
                                    return None, None, pd.DataFrame({"Message": ["No analytics data available yet"]})
                                
                                df = pd.read_excel(feedback_file)
                                
                                if df.empty:
                                    return None, None, pd.DataFrame({"Message": ["No analytics data available yet"]})
                                
                                # Calculate statistics by agent/model
                                model_stats = df.groupby('Agent')[['Response Quality', 'Information Accuracy', 'Response Time', 'Sources Used']].mean().reset_index()
                                model_stats['Total Queries'] = df.groupby('Agent').size().values
                                
                                # Create response time chart
                                import matplotlib.pyplot as plt
                                
                                fig1, ax1 = plt.subplots(figsize=(10, 6))
                                model_stats.plot(x='Agent', y='Response Time', kind='bar', ax=ax1, color='skyblue')
                                ax1.set_title('Average Response Time by Model')
                                ax1.set_ylabel('Seconds')
                                ax1.set_xlabel('')
                                plt.tight_layout()
                                
                                # Create sources chart
                                fig2, ax2 = plt.subplots(figsize=(10, 6))
                                model_stats.plot(x='Agent', y='Sources Used', kind='bar', ax=ax2, color='lightgreen')
                                ax2.set_title('Average Number of Sources Used by Model')
                                ax2.set_ylabel('Count')
                                ax2.set_xlabel('')
                                plt.tight_layout()
                                
                                return fig1, fig2, model_stats
                            except Exception as e:
                                logger.error(f"Error refreshing analytics: {str(e)}")
                                return None, None, pd.DataFrame({"Error": [f"Failed to load analytics: {str(e)}"]})
                        
                        refresh_btn.click(refresh_analytics, outputs=[performance_chart, source_chart, crew_stats])

        return demo

    except Exception as e:
        logger.error(f"Error creating Gradio interface: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def sanitize_text(text):
    """Remove or replace problematic Unicode characters."""
    if not isinstance(text, str):
        return str(text)
    
    # Replace common problematic characters with ASCII equivalents
    replacements = {
        '\u2264': '<=',  # ≤
        '\u2265': '>=',  # ≥
        '\u2022': '*',   # bullet point
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2013': '-',   # en dash
        '\u2014': '--',  # em dash
        '\u00b0': ' degrees', # degree symbol
        '\u00b2': '^2',  # squared
        '\u00b3': '^3',  # cubed
        '\u00a9': '(c)', # copyright
        '\u00ae': '(R)', # registered trademark
        '\u2122': '(TM)' # trademark
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # For any remaining problematic characters, replace with a closest ASCII equivalent
    # or remove if no good replacement exists
    return text.encode('ascii', errors='replace').decode('ascii')        

# Add new function for web document uploads
async def upload_web_pdf_to_qdrant(pdf_content: bytes, pdf_info: Dict) -> Dict:
    """Upload web PDF document to Qdrant web_docs collection."""
    try:
        logger.info(f"Starting Qdrant upload for web PDF: {pdf_info.get('title')}")
        
        # Check if PDF already exists in database
        pdf_url = pdf_info.get('url')
        if await is_pdf_in_qdrant(pdf_url):
            return {
                "status": "skipped",
                "message": f"PDF {pdf_info.get('title')} already exists in database",
                "url": pdf_url
            }

        # Extract text from PDF content
        text = extract_text_from_pdf(pdf_content)
        if not text:
            return {
                "status": "skipped", 
                "message": f"No content could be extracted from {pdf_info.get('title')}",
                "url": pdf_info.get('url')
            }

        # Generate embeddings for text
        text_embedding = embedding_model.encode(text)
        
        # Create point with metadata
        point = {
            "id": str(uuid.uuid4()),
            "vector": text_embedding.tolist(),
            "payload": {
                "metadata_storage_name": pdf_info.get('title'),
                "metadata_storage_path": pdf_info.get('url'),
                "content": text,
                "url": pdf_info.get('url'),
                "context": pdf_info.get('context', ''),
                "upload_time": datetime.now().isoformat(),
                "source": "web"
            }
        }

        # Upload to Qdrant web_docs collection
        collection_name = "web_docs"
        qdrant_client = get_qdrant_client()
      
        # Create collection if it doesn't exist
        try:
                qdrant_client.get_collection(collection_name)
        except Exception:
                qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True
            )

        return {
            "status": "success",
            "message": f"Successfully uploaded {pdf_info.get('title')} to Qdrant web_docs",
            "document_id": point["id"]
            }

    except Exception as e:
        error_msg = f"Qdrant web PDF upload error: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "url": pdf_info.get('url')
        }

def launch_gradio():
    """Launch the Gradio interface with fallback mechanisms."""
    try:
        demo = create_gradio_interface()
        if demo:
            # Try different ports if the default one is in use
            ports_to_try = [8080, 7860, 3000, 0]  # 0 lets OS choose port
            
            for port in ports_to_try:
                try:
                    logger.info(f"Attempting to launch Gradio on port {port}")
                    demo.launch(
                        server_name="127.0.0.1",  # Use localhost for better stability
                        server_port=port,
                        share=True,
                        auth=None,  # Add authentication if needed
                        prevent_thread_lock=False,  # Changed to False to block main thread
                        show_error=True,
                        quiet=False
                    )
                    logger.info(f"Gradio successfully launched on port {port}")
                    return
                except Exception as e:
                    if "port is already in use" in str(e).lower():
                        logger.warning(f"Port {port} is in use, trying next port")
                        continue
                    else:
                        logger.error(f"Error launching Gradio on port {port}: {str(e)}")
                        raise
            
            logger.error("All ports are in use. Please free up a port and try again.")
            
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {str(e)}")
        # Fallback to FastAPI
        try:
            port = find_available_port()
            logger.info(f"Falling back to FastAPI on port {port}")
            import uvicorn
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=port,
                log_level="info"
            )
        except Exception as e2:
            logger.error(f"Failed to launch FastAPI fallback: {str(e2)}")
            raise RuntimeError("Could not start server. Please check if ports are available and try again.")

def find_available_port() -> int:
    """Let the OS find an available port by binding to port 0."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
            return port
    except Exception as e:
        logger.error(f"Error finding available port: {str(e)}")
        # Return a random port in the higher range as fallback
        return random.randint(49152, 65535)

if __name__ == "__main__":
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Load environment variables
        load_dotenv()
        
        # Create and launch Gradio interface
        launch_gradio()
        
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        print(f"Error starting application: {str(e)}")
        
        # Try to kill any existing processes on the ports
        try:
            import psutil
            for port in [8080, 7860, 3000]:
                for proc in psutil.process_iter(['pid', 'name', 'connections']):
                    try:
                        for conn in proc.connections():
                            if conn.laddr.port == port:
                                logger.info(f"Terminating process using port {port}: {proc.pid}")
                                psutil.Process(proc.pid).terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except Exception as e:
            logger.error(f"Error cleaning up processes: {str(e)}")
