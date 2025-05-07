"""
Utility modules for the application.
"""

from .supabase import SupabaseClient

def get_supabase_client():
    """Get the Supabase client instance."""
    return SupabaseClient() 