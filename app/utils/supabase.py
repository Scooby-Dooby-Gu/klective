from typing import List, Optional, Dict, Any
from uuid import UUID
from supabase import create_client, Client, ClientOptions
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

from app.models.pydantic_models import (
    Collection,
    Document,
    TimelineEvent,
    DocumentEmbedding,
    ProcessingResult,
    TextChunk
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SupabaseClient:
    def __init__(self):
        """Initialize Supabase client."""
        # Load environment variables
        load_dotenv()
        
        # Get Supabase configuration
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("Missing Supabase URL or key in environment variables")
        
        # Initialize Supabase client
        self.client = create_client(self.url, self.key)
        self.storage = self.client.storage
        
        # Ensure the documents bucket exists
        try:
            self.storage.get_bucket("documents")
        except Exception as e:
            logger.warning(f"Documents bucket not found, creating it: {e}")
            self.storage.create_bucket("documents", {"public": True})
        
        logger.info("Supabase client initialized successfully")
        
        # Initialize database tables
        self.collections = self.client.table("collections")
        self.documents = self.client.table("documents")
        self.events = self.client.table("timeline_events")
        self.embeddings = self.client.table("document_embeddings")
        
        # For development/testing, use a default user ID
        self.default_user_id = "test_user_id"
    
    def get_collections(self, user_id: UUID) -> List[Collection]:
        """Fetch collections for a user."""
        try:
            response = self.collections.select("*").eq("user_id", str(user_id)).execute()
            return [Collection(**item) for item in response.data]
        except Exception as e:
            print(f"Error fetching collections: {str(e)}")
            return []
    
    def create_collection(self, name: str, description: Optional[str], user_id: UUID) -> Collection:
        """Create a new collection."""
        try:
            response = self.collections.insert({
                "name": name,
                "description": description,
                "user_id": str(user_id)
            }).execute()
            return Collection(**response.data[0])
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise
    
    def get_documents(self, collection_id: Optional[str] = None, user_id: Optional[UUID] = None) -> List[Document]:
        """Fetch documents, optionally filtered by collection and user."""
        try:
            query = self.documents.select("*")
            if collection_id:
                query = query.eq("collection_id", collection_id)
            if user_id:
                query = query.eq("user_id", str(user_id))
            response = query.execute()
            
            # Ensure collection_id is not None for each document
            documents = []
            for item in response.data:
                if item.get("collection_id") is None:
                    logger.warning(f"Document {item.get('id')} has no collection_id")
                    continue
                documents.append(Document(**item))
            return documents
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []
    
    def upload_document(self, collection_id: str, file_path: str, file_name: str, mime_type: str, user_id: UUID) -> Document:
        """Upload a document to storage and create a document record."""
        try:
            # Read file data
            with open(file_path, "rb") as f:
                file_data = f.read()
            
            # Create storage path with timestamp to prevent duplicates
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name, ext = os.path.splitext(file_name)
            unique_file_name = f"{base_name}_{timestamp}{ext}"
            
            # Use the correct bucket name and path structure
            bucket_name = "documents"
            storage_path = f"{str(user_id)}/{unique_file_name}"
            
            print(f"Uploading file to bucket: {bucket_name}, path: {storage_path}")
            print(f"File size: {len(file_data)} bytes")
            print(f"MIME type: {mime_type}")
            
            # Upload to storage with proper content type
            try:
                upload_result = self.storage.from_(bucket_name).upload(
                    storage_path, 
                    file_data,
                    {"contentType": mime_type}
                )
                print(f"Upload result: {upload_result}")
            except Exception as upload_error:
                print(f"Error during storage upload: {str(upload_error)}")
                raise
            
            # Verify the file exists in storage
            try:
                # List all files in the user's directory
                files = self.storage.from_(bucket_name).list(str(user_id))
                print(f"Files in storage directory: {files}")
                
                # Check if our file exists
                file_exists = any(f["name"] == unique_file_name for f in files)
                print(f"File exists in storage: {file_exists}")
                
                if not file_exists:
                    raise Exception(f"File {unique_file_name} not found in storage after upload")
            except Exception as list_error:
                print(f"Error checking storage: {str(list_error)}")
                raise
            
            # Create document record with the correct storage path
            try:
                response = self.documents.insert({
                    "collection_id": str(collection_id),
                    "file_name": file_name,  # Original file name
                    "file_path": storage_path,  # Use the actual storage path
                    "file_size": len(file_data),
                    "mime_type": mime_type,
                    "user_id": str(user_id)
                }).execute()
                print(f"Document record created: {response.data[0]}")
                return Document(**response.data[0])
            except Exception as db_error:
                print(f"Error creating document record: {str(db_error)}")
                raise
                
        except Exception as e:
            print(f"Error uploading document: {str(e)}")
            raise
    
    async def save_processing_result(self, document_id: str, result: ProcessingResult) -> None:
        """Save document processing results."""
        try:
            logger.info(f"Saving processing result for document {document_id}")
            
            # Update document status
            logger.debug("Updating document status...")
            update_data = {
                "processed": True,
                "processed_at": datetime.now().isoformat(),  # Use current time instead of result.processed_at
                "summary": result.summary,
                "extraction_metadata": {
                    "success": result.success,
                    "error": result.error,
                    "events_count": len(result.events),
                    "embeddings_count": len(result.embeddings)
                }
            }
            
            # Try to add description if the column exists
            try:
                update_data["description"] = f"Processed document with {len(result.embeddings)} vector chunks and {len(result.events)} timeline events"
            except Exception as e:
                logger.warning(f"Could not add description to document update: {e}")
            
            # Update document status
            self.documents.update(update_data).eq("id", str(document_id)).execute()
            logger.debug("Document status updated successfully")
            
            # Delete existing embeddings before saving new ones
            logger.debug(f"Deleting existing embeddings for document {document_id}...")
            await self.delete_document_embeddings(document_id)
            
            # Save embeddings
            logger.debug(f"Saving {len(result.embeddings)} embeddings...")
            if result.embeddings:
                embeddings_data = []
                for embedding in result.embeddings:
                    # Convert element_metadata to a serializable format
                    element_metadata = embedding.element_metadata
                    if element_metadata and hasattr(element_metadata, 'to_dict'):
                        element_metadata = element_metadata.to_dict()
                    
                    embedding_data = {
                        "document_id": str(document_id),
                        "chunk_index": embedding.chunk_index,
                        "content": embedding.content,
                        "embedding": embedding.embedding,
                        "source_element_id": embedding.source_element_id,
                        "element_type": embedding.element_type,
                        "element_metadata": element_metadata
                    }
                    embeddings_data.append(embedding_data)
                
                # Insert embeddings without await
                self.embeddings.insert(embeddings_data).execute()
                logger.debug("Embeddings saved successfully")
            
            # Save events
            logger.debug(f"Saving {len(result.events)} events...")
            if result.events:
                for event in result.events:
                    logger.debug(f"Saving event: {event}")
                    try:
                        # Insert event without await
                        self.events.insert({
                            "document_id": str(document_id),
                            "event_date": event.event_date.isoformat(),
                            "title": event.title,
                            "description": event.description,
                            "importance": event.importance,
                            "category": event.category,
                            "actors": event.actors,
                            "location": event.location,
                            "references": event.references,
                            "page_or_section_references": event.page_or_section_references,
                            "confidence_score": event.confidence_score
                        }).execute()
                        logger.debug("Event saved successfully")
                    except Exception as e:
                        logger.error(f"Error saving event: {str(e)}")
                        logger.error(f"Event data: {event}")
                        raise
                logger.info("All events saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving processing results: {str(e)}")
            raise
    
    def get_events(self, document_id: str) -> List[TimelineEvent]:
        """Fetch timeline events for a document."""
        try:
            response = self.events.select("*").eq("document_id", document_id).execute()
            return [TimelineEvent(**item) for item in response.data]
        except Exception as e:
            print(f"Error fetching events: {str(e)}")
            return []
    
    def search_documents(self, query: str, user_id: UUID) -> List[Document]:
        """Search documents using vector similarity."""
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Search embeddings
            response = self.embeddings.select(
                "document_id",
                "content",
                "similarity"
            ).eq("user_id", str(user_id)).order(
                "similarity",
                desc=True
            ).limit(10).execute()
            
            # Get unique document IDs
            doc_ids = list(set(item["document_id"] for item in response.data))
            
            # Fetch full document details
            documents = []
            for doc_id in doc_ids:
                doc_response = self.documents.select("*").eq("id", doc_id).execute()
                if doc_response.data:
                    documents.append(Document(**doc_response.data[0]))
            
            return documents
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API."""
        # This should be implemented using your preferred embedding model
        # For now, return a dummy embedding
        return [0.0] * 1536 

    async def get_document(self, doc_id: str) -> Document:
        """Get a single document by its ID."""
        try:
            response = self.client.table("documents").select("*").eq("id", doc_id).single().execute()
            if response.data:
                document = Document(**response.data)
                print(f"Retrieved document with path: {document.file_path}")
                
                # Verify the file exists in storage
                try:
                    files = self.storage.from_("documents").list(os.path.dirname(document.file_path))
                    print(f"Files in storage directory: {files}")
                    
                    if not any(f["name"] == os.path.basename(document.file_path) for f in files):
                        print(f"Warning: File {document.file_path} not found in storage")
                except Exception as e:
                    print(f"Error checking storage: {str(e)}")
                
                return document
            return None
        except Exception as e:
            print(f"Error getting document: {str(e)}")
            return None

    def delete_document(self, doc_id: str, user_id: UUID) -> bool:
        """Delete a document and its associated storage file."""
        try:
            # Get the document first to get the file path
            response = self.documents.select("*").eq("id", doc_id).eq("user_id", str(user_id)).single().execute()
            if not response.data:
                print(f"Document {doc_id} not found or not owned by user")
                return False
                
            document = Document(**response.data)
            
            # Delete timeline events first
            try:
                self.events.delete().eq("document_id", doc_id).execute()
                print(f"Deleted timeline events for document: {doc_id}")
            except Exception as events_error:
                print(f"Error deleting timeline events: {str(events_error)}")
                return False
            
            # Delete embeddings
            try:
                self.embeddings.delete().eq("document_id", doc_id).execute()
                print(f"Deleted embeddings for document: {doc_id}")
            except Exception as embeddings_error:
                print(f"Error deleting embeddings: {str(embeddings_error)}")
                return False
            
            # Delete from storage
            try:
                self.storage.from_("documents").remove([document.file_path])
                print(f"Deleted file from storage: {document.file_path}")
            except Exception as storage_error:
                print(f"Error deleting from storage: {str(storage_error)}")
                # Continue with database deletion even if storage deletion fails
            
            # Delete from database
            self.documents.delete().eq("id", doc_id).eq("user_id", str(user_id)).execute()
            print(f"Deleted document record: {doc_id}")
            
            return True
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False

    async def save_embeddings(self, document_id: str, embeddings: List[DocumentEmbedding]) -> None:
        """Save document embeddings to the database."""
        try:
            if not embeddings:
                logger.warning("No embeddings to save")
                return
                
            # Convert embeddings to list of dictionaries
            embeddings_data = [embedding.model_dump() for embedding in embeddings]
            
            # Insert embeddings into the database
            response = self.embeddings.insert(embeddings_data).execute()
            
            if response.error:
                raise Exception(f"Error saving embeddings: {response.error}")
                
            logger.info(f"Successfully saved {len(embeddings_data)} embeddings for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise

    async def save_timeline_event(self, event_data: Dict[str, Any]) -> None:
        """Save a single timeline event to Supabase."""
        try:
            # Ensure required fields are present
            required_fields = ["document_id", "event_date", "title", "description"]
            for field in required_fields:
                if field not in event_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Convert event_date to ISO format string if it's a datetime object
            if hasattr(event_data["event_date"], "isoformat"):
                event_data["event_date"] = event_data["event_date"].isoformat()
            
            # Insert the event into the timeline_events table
            response = self.client.table("timeline_events").insert(event_data).execute()
            
            # Check if the response has data
            if not response.data:
                raise Exception(f"Error saving timeline event: {response}")
                
            logger.debug(f"Successfully saved timeline event for document {event_data['document_id']}")
        except Exception as e:
            logger.error(f"Error saving timeline event: {str(e)}")
            raise

    async def update_document_status(self, doc_id: str, processed: bool, processed_at: datetime, summary: str) -> None:
        """Update document processing status."""
        try:
            self.client.table("documents").update({
                "processed": processed,
                "processed_at": processed_at.isoformat(),
                "summary": summary
            }).eq("id", doc_id).execute()
        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")
            raise

    async def get_document_content(self, doc_id: str) -> str:
        """Get the processed content of a document."""
        try:
            # Get the document embeddings which contain the content
            response = self.client.table("document_embeddings").select("content").eq("document_id", doc_id).execute()
            
            if not response.data:
                return ""
            
            # Combine all content chunks into a single string
            content = " ".join(item["content"] for item in response.data if item.get("content"))
            return content
            
        except Exception as e:
            print(f"Error getting document content: {str(e)}")
            return ""

    async def delete_document_embeddings(self, document_id: str) -> None:
        """Delete all embeddings for a document."""
        try:
            logger.info(f"Deleting embeddings for document {document_id}")
            self.embeddings.delete().eq("document_id", document_id).execute()
            logger.info("Embeddings deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting embeddings: {str(e)}")
            raise

    async def delete_timeline_events(self, document_id: str) -> None:
        """Delete all timeline events for a document."""
        try:
            logger.info(f"Deleting timeline events for document {document_id}")
            self.events.delete().eq("document_id", document_id).execute()
            logger.info("Timeline events deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting timeline events: {str(e)}")
            raise

    async def save_api_calls(self, document_id: UUID, api_calls: int, api_type: str):
        """Save API call count for a document."""
        try:
            data = {
                "document_id": str(document_id),
                "api_calls": api_calls,
                "api_type": api_type,  # 'document' or 'timeline'
                "timestamp": datetime.now().isoformat()
            }
            self.client.table("api_calls").insert(data).execute()
        except Exception as e:
            logger.error(f"Error saving API calls: {str(e)}")
            raise

    async def get_api_calls(self, document_id: Optional[UUID] = None) -> List[Dict]:
        """Get API call counts for documents."""
        try:
            query = self.client.table("api_calls").select("*")
            if document_id:
                query = query.eq("document_id", str(document_id))
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting API calls: {str(e)}")
            return []

    async def get_total_api_calls(self) -> Dict[str, int]:
        """Get total API calls by type."""
        try:
            response = self.client.table("api_calls").select("api_type", "api_calls").execute()
            totals = {"document": 0, "timeline": 0}
            for record in response.data:
                totals[record["api_type"]] += record["api_calls"]
            return totals
        except Exception as e:
            logger.error(f"Error getting total API calls: {str(e)}")
            return {"document": 0, "timeline": 0} 