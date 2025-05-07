from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
import logging
import json
from openai import OpenAI
import os
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.models.pydantic_models import TimelineEvent
from app.utils.supabase import SupabaseClient

# Configure logger
logger = logging.getLogger(__name__)

class TimelineExtractorAgent:
    """Agent responsible for extracting timeline events from stored documents using GPT-4o."""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.supabase_client = SupabaseClient()
        self.timeline_cache_dir = Path("data/timeline_cache")
        self.timeline_cache_dir.mkdir(parents=True, exist_ok=True)
        self.processing_interval = 300  # 5 minutes in seconds
        self.last_check_time = None
        self.is_running = False
        self.background_task = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,  # Increased chunk size
            chunk_overlap=200,  # Overlap to avoid missing events at chunk boundaries
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    async def start_background_processing(self):
        """Start the background task to process new documents."""
        if self.is_running:
            logger.warning("Background processing is already running")
            return
        
        self.is_running = True
        self.background_task = asyncio.create_task(self._background_processing())
        logger.info("Started background processing for timeline extraction")
    
    async def stop_background_processing(self):
        """Stop the background processing task."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped background processing for timeline extraction")
    
    async def _background_processing(self):
        """Background task that periodically checks for new documents to process."""
        while self.is_running:
            try:
                await self._process_new_documents()
                await asyncio.sleep(self.processing_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background processing: {str(e)}")
                await asyncio.sleep(self.processing_interval)
    
    async def _process_new_documents(self):
        """Process any new documents that haven't been processed for timeline events."""
        try:
            # Get all processed documents
            response = self.supabase_client.documents.select("*").eq("processed", True).execute()
            if not response.data:
                return
            
            # Get all documents that have timeline events
            events_response = self.supabase_client.events.select("document_id").execute()
            processed_doc_ids = {event["document_id"] for event in events_response.data}
            
            # Find documents that need processing
            for doc in response.data:
                if doc["id"] not in processed_doc_ids:
                    logger.info(f"Found new document to process: {doc['id']}")
                    try:
                        events = await self.extract_events_from_document(doc["id"])
                        if events:
                            logger.info(f"Successfully extracted {len(events)} events for document {doc['id']}")
                        else:
                            logger.warning(f"No events extracted for document {doc['id']}")
                    except Exception as e:
                        logger.error(f"Error processing document {doc['id']}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error checking for new documents: {str(e)}")
    
    def _get_cache_path(self, document_id: str) -> Path:
        """Get the path for caching timeline events for a document."""
        return self.timeline_cache_dir / f"{document_id}_timeline.json"
    
    def save_events_locally(self, document_id: UUID, events: List[TimelineEvent]):
        """Save events to a local JSON file for debugging."""
        try:
            events_data = {
                "document_id": str(document_id),
                "events": [
                    {
                        "event_date": event.event_date.isoformat() if event.event_date else None,
                        "title": event.title,
                        "description": event.description,
                        "importance": event.importance,
                        "category": event.category,
                        "actors": event.actors,
                        "location": event.location,
                        "confidence_score": event.confidence_score
                    }
                    for event in events
                ]
            }
            
            os.makedirs("debug", exist_ok=True)
            with open(f"debug/timeline_events_{document_id}.json", "w") as f:
                json.dump(events_data, f, indent=2)
            
            logger.info(f"Saved {len(events)} events to debug/timeline_events_{document_id}.json")
        except Exception as e:
            logger.error(f"Error saving timeline events locally: {str(e)}")
            raise
    
    def _load_events_from_cache(self, document_id: UUID) -> Optional[List[TimelineEvent]]:
        """Load timeline events from local cache."""
        try:
            cache_path = self._get_cache_path(str(document_id))
            if not cache_path.exists():
                return None
            
            with open(cache_path) as f:
                data = json.load(f)
                
            events = []
            for event_data in data["events"]:
                event = TimelineEvent(
                    document_id=document_id,
                    event_date=datetime.fromisoformat(event_data["event_date"]) if event_data.get("event_date") else None,
                    title=event_data["title"],
                    description=event_data["description"],
                    importance=event_data["importance"],
                    category=event_data.get("category"),
                    actors=event_data.get("actors", []),
                    location=event_data.get("location"),
                    confidence_score=event_data.get("confidence_score", 0.5)
                )
                events.append(event)
                
            return events
        except Exception as e:
            logger.error(f"Error loading timeline events from cache: {str(e)}")
            return None
    
    async def save_events_to_supabase(self, document_id: UUID, events: List[TimelineEvent]):
        """Save timeline events to Supabase."""
        try:
            logger.info(f"Saving {len(events)} timeline events to Supabase")
            
            # Delete existing events for this document
            await self.supabase_client.delete_timeline_events(document_id)
            
            # Save each event
            for event in events:
                event_data = {
                    "document_id": str(document_id),
                    "event_date": event.event_date.isoformat() if event.event_date else None,
                    "title": event.title,
                    "description": event.description,
                    "importance": event.importance,
                    "category": event.category,
                    "actors": event.actors,
                    "location": event.location,
                    "confidence_score": event.confidence_score
                }
                await self.supabase_client.save_timeline_event(event_data)
                
            logger.info("Successfully saved timeline events to Supabase")
        except Exception as e:
            logger.error(f"Error saving timeline events to Supabase: {str(e)}")
            raise
    
    async def extract_events_from_document(self, document_id: UUID) -> List[TimelineEvent]:
        """Extract timeline events from a document."""
        try:
            logger.info(f"Extracting timeline events from document {document_id}")
            
            # Get document content
            document = await self.supabase_client.get_document(document_id)
            if not document:
                logger.error(f"Document {document_id} not found")
                return []
            
            # Get document content
            content = await self.supabase_client.get_document_content(document_id)
            if not content:
                logger.error(f"No content found for document {document_id}")
                return []
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Process each chunk
            all_events = []
            api_calls = 0
            for chunk in chunks:
                events = await self._extract_events_with_openai(chunk, document_id)
                all_events.extend(events)
                api_calls += 1  # Count each chunk processing
            
            # Deduplicate events
            seen_events = set()
            unique_events = []
            
            for event in all_events:
                event_key = (event.title, event.event_date.isoformat() if event.event_date else None)
                if event_key not in seen_events:
                    seen_events.add(event_key)
                    unique_events.append(event)
            
            # Sort events by date
            unique_events.sort(key=lambda x: x.event_date if x.event_date else datetime.min)
            
            # Save events locally and to Supabase
            self.save_events_locally(document_id, unique_events)
            await self.save_events_to_supabase(document_id, unique_events)
            
            # Save API call count
            await self.supabase_client.save_api_calls(document_id, api_calls, "timeline")
            
            logger.info(f"Extracted {len(unique_events)} unique events from document {document_id} using {api_calls} API calls")
            
            return unique_events
            
        except Exception as e:
            logger.error(f"Error extracting timeline events: {str(e)}")
            return []
    
    async def _extract_events_with_openai(self, chunk: str, document_id: str) -> List[Dict]:
        """Extract events from a chunk of text using OpenAI."""
        try:
            logger.info("Making OpenAI API call to extract events")
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Extract all chronological events from the following text. 
                        Return a JSON array of events, where each event has:
                        - event_date (ISO format date string, e.g. "1957-10-21" for October 21, 1957)
                        - title (string)
                        - description (string)
                        - importance (integer 1-5)
                        - category (string, optional)
                        - actors (array of strings, optional)
                        - location (string, optional)
                        - confidence_score (number 0-1, optional)
                        
                        Example format:
                        [
                            {{
                                "event_date": "1957-10-21",
                                "title": "Birth of Wolfgang Ketterle",
                                "description": "Wolfgang Ketterle was born in Heidelberg, Germany",
                                "importance": 5,
                                "category": "Life Event",
                                "actors": ["Wolfgang Ketterle"],
                                "location": "Heidelberg, Germany",
                                "confidence_score": 1.0
                            }},
                            {{
                                "event_date": "1960-01-01",
                                "title": "Family moved to Eppelheim",
                                "description": "When Wolfgang Ketterle was three, his family moved from Heidelberg to Eppelheim",
                                "importance": 3,
                                "category": "Life Event",
                                "actors": ["Wolfgang Ketterle", "Family"],
                                "location": "Eppelheim, Germany",
                                "confidence_score": 0.9
                            }}
                        ]
                        
                        Important: 
                        1. Use actual dates from the text, not placeholder dates
                        2. If only a year is mentioned, use January 1st of that year
                        3. If a month and year are mentioned, use the 1st of that month
                        4. If a specific date is mentioned, use that exact date
                        5. Do not use timestamps or time components, just the date
                        6. Only create events for the main characters mentioned multiple times in the text. Do not create events for one time mentions.
                        
                        Text to analyze:
                        {chunk}"""
                    }
                ],
                temperature=0.7,
            )

            # Log the raw response for debugging
            logger.debug(f"OpenAI API response: {response}")

            # Parse the response
            content = response.choices[0].message.content
            logger.debug(f"Response content: {content}")

            # Remove markdown code block markers if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Try to parse the JSON
            try:
                events = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Raw content: {content}")
                return []

            # Validate the response format
            if not isinstance(events, list):
                logger.error(f"Expected list of events, got {type(events)}")
                return []

            # Ensure importance is an integer
            for event in events:
                if "importance" in event:
                    event["importance"] = int(round(float(event["importance"])))

            # Add document_id to each event
            for event in events:
                event["document_id"] = document_id

            logger.info(f"Successfully extracted {len(events)} events")
            return events

        except Exception as e:
            logger.error(f"Error extracting events from chunk: {str(e)}")
            logger.error(f"Chunk content: {chunk}")
            return []
    
    async def extract_events_from_collection(self, collection_id: UUID) -> List[TimelineEvent]:
        """Extract timeline events from all documents in a collection."""
        try:
            # Get all documents in the collection
            documents = await self.supabase_client.get_documents_by_collection(collection_id)
            if not documents:
                logger.warning(f"No documents found in collection {collection_id}")
                return []
            
            # Extract events from each document
            all_events = []
            for document in documents:
                events = await self.extract_events_from_document(document.id)
                all_events.extend(events)
            
            # Sort events by date
            return sorted(all_events, key=lambda x: x.event_date)
            
        except Exception as e:
            logger.error(f"Error in extract_events_from_collection: {str(e)}")
            return [] 