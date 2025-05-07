"""
Agent modules for document processing and timeline generation.
"""

from typing import Optional
import os
from dotenv import load_dotenv

from .processor_agent import DocumentProcessorAgent
from .timeline_agent import TimelineGeneratorAgent
from app.utils.supabase import SupabaseClient

# Load environment variables
load_dotenv()

# Initialize global agent instances
_processor_agent: Optional[DocumentProcessorAgent] = None
_timeline_agent: Optional[TimelineGeneratorAgent] = None
_supabase_client: Optional[SupabaseClient] = None

def get_processor_agent() -> DocumentProcessorAgent:
    """Get or create the DocumentProcessorAgent instance"""
    global _processor_agent
    if _processor_agent is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _processor_agent = DocumentProcessorAgent(openai_api_key)
    return _processor_agent

def get_timeline_agent() -> TimelineGeneratorAgent:
    """Get or create the TimelineGeneratorAgent instance"""
    global _timeline_agent
    if _timeline_agent is None:
        _timeline_agent = TimelineGeneratorAgent()
    return _timeline_agent

def get_supabase_client() -> SupabaseClient:
    """Get or create the SupabaseClient instance"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseClient()
    return _supabase_client

def initialize_agents() -> None:
    """Initialize all agents and verify environment variables"""
    # Verify required environment variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for document processing and embeddings",
        "SUPABASE_URL": "Supabase project URL",
        "SUPABASE_KEY": "Supabase API key"
    }
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please set these variables in your .env file or environment."
        )
    
    # Initialize agents
    get_processor_agent()
    get_timeline_agent()
    get_supabase_client() 