from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
import tempfile
import os
import aiohttp
from pydantic import BaseModel, Field, ConfigDict
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from unstructured.partition.docx import partition_docx
from unstructured.staging.base import elements_to_json
from langchain.schema import HumanMessage
import logging
import json
import uuid

from app.models.pydantic_models import (
    Document,
    DocumentEmbedding,
    ProcessingResult,
    TextChunk
)
from app.utils.supabase import SupabaseClient
from app.agents.timeline_extractor_agent import TimelineExtractorAgent

# Configure logger
logger = logging.getLogger(__name__)

class DocumentProcessorAgent(BaseModel):
    """Agent responsible for processing documents using Unstructured"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    openai_api_key: str
    embeddings_model: OpenAIEmbeddings
    chat_model: ChatOpenAI
    text_splitter: RecursiveCharacterTextSplitter
    supabase_client: SupabaseClient
    
    def __init__(self, openai_api_key: str):
        super().__init__(
            openai_api_key=openai_api_key,
            embeddings_model=OpenAIEmbeddings(openai_api_key=openai_api_key),
            chat_model=ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o"),
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
            ),
            supabase_client=SupabaseClient()
        )
    
    async def download_document(self, file_path: str) -> str:
        """Download a document from Supabase storage."""
        try:
            logger.info(f"Downloading document from path: {file_path}")
            
            # Use the Supabase storage client to download the file
            data = self.supabase_client.storage.from_("documents").download(file_path)
            
            if not data:
                raise ValueError(f"Failed to download document: No data returned")
                
            # Create a temporary file to save the document
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1])
            temp_file.write(data)
            temp_file.close()
            
            # Verify the file exists and log its size
            if os.path.exists(temp_file.name):
                file_size = os.path.getsize(temp_file.name)
                logger.info(f"Document downloaded successfully to: {temp_file.name} (size: {file_size} bytes)")
            else:
                raise FileNotFoundError(f"Temporary file was not created: {temp_file.name}")
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise
    
    async def process_document(self, document_id: str) -> ProcessingResult:
        """Process a document and generate embeddings"""
        local_path = None
        api_calls = 0
        try:
            # Get document from Supabase
            document = await self.supabase_client.get_document(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Download document from storage
            local_path = await self.download_document(document.file_path)
            
            # Verify the file exists before processing
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Downloaded file not found at: {local_path}")
            logger.info(f"Verifying file before processing - exists: {os.path.exists(local_path)}, size: {os.path.getsize(local_path)} bytes")
            
            # Extract content
            elements = await self.extract_content(local_path, document.mime_type)
            if not elements:
                raise ValueError("No content extracted from document")
            
            # Generate summary
            summary = await self.generate_summary(elements)
            api_calls += 1  # Count summary generation
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(elements, document_id)
            api_calls += len(embeddings)  # Count each embedding generation
            
            # Save processing result
            result = ProcessingResult(
                document_id=document_id,
                success=True,
                summary=summary,
                embeddings=embeddings,
                events=[]  # Timeline events will be handled by the TimelineExtractorAgent
            )
            
            # Update document status
            await self.supabase_client.save_processing_result(document_id, result)
            
            # Save API call count
            await self.supabase_client.save_api_calls(document_id, api_calls, "document")
            
            logger.info(f"Document processing completed successfully. Embeddings: {len(embeddings)}, API calls: {api_calls}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
        finally:
            # Clean up the temporary file
            if local_path and os.path.exists(local_path):
                try:
                    os.unlink(local_path)
                    logger.info(f"Temporary file {local_path} cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {local_path}: {e}")
    
    async def extract_content(self, file_path: str, mime_type: str) -> List[Any]:
        """Extract content from document using Unstructured"""
        try:
            # Use the appropriate partition function based on mime type
            if mime_type == "application/pdf":
                elements = partition_pdf(file_path)
            elif mime_type == "text/plain":
                elements = partition_text(file_path)
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                elements = partition_docx(file_path)
            else:
                elements = partition(file_path)
            
            # Convert elements to a consistent format
            processed_elements = []
            for i, element in enumerate(elements):
                if hasattr(element, "text"):
                    processed_elements.append({
                        "element_id": str(i),  # Use index as element ID
                        "type": type(element).__name__,
                        "text": element.text,
                        "metadata": element.metadata if hasattr(element, "metadata") else {}
                    })
                else:
                    processed_elements.append({
                        "element_id": str(i),  # Use index as element ID
                        "type": "Unknown",
                        "text": str(element),
                        "metadata": {}
                    })
            
            return processed_elements
            
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            raise
    
    async def generate_summary(self, elements: List[Any]) -> str:
        """Generate a summary of the document content."""
        try:
            # Extract text content from elements
            text_content = []
            for element in elements:
                if isinstance(element, dict):
                    text = element.get("text", "")
                else:
                    text = getattr(element, "text", "")
                if text:
                    text_content.append(text)
            
            if not text_content:
                logger.warning("No text content found in elements for summary generation")
                return ""
            
            # Join all text content
            full_text = "\n".join(text_content)
            
            # Split text into chunks using RecursiveCharacterTextSplitter
            chunks = self.text_splitter.split_text(full_text)
            
            # Process each chunk and combine summaries
            chunk_summaries = []
            for chunk in chunks:
                prompt = HumanMessage(content=f"""
                Please provide a concise summary of the following text chunk:
                
                {chunk}
                
                Focus on the main points and key information. The summary should be clear and informative.
                """)
                
                response = await self.chat_model.agenerate([[prompt]])
                chunk_summaries.append(response.generations[0][0].text.strip())
            
            # Recursively combine summaries in batches to avoid context length issues
            async def combine_summaries(summaries: List[str], batch_size: int = 3) -> str:
                if len(summaries) <= batch_size:
                    # If we have a small enough batch, combine them directly
                    combined_text = "\n".join(summaries)
                    prompt = HumanMessage(content=f"""
                    Please provide a concise final summary that combines the following summaries:
                    
                    {combined_text}
                    
                    Focus on the main points and key information. The summary should be clear and informative.
                    """)
                    
                    response = await self.chat_model.agenerate([[prompt]])
                    return response.generations[0][0].text.strip()
                else:
                    # Split into batches and recursively combine
                    batches = [summaries[i:i + batch_size] for i in range(0, len(summaries), batch_size)]
                    batch_summaries = []
                    for batch in batches:
                        batch_summary = await combine_summaries(batch, batch_size)
                        batch_summaries.append(batch_summary)
                    return await combine_summaries(batch_summaries, batch_size)
            
            # Start the recursive combination process
            final_summary = await combine_summaries(chunk_summaries)
            return final_summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise
    
    def create_chunks(self, elements: List[Any]) -> List[TextChunk]:
        """Process elements into semantic chunks"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for element in elements:
            if not hasattr(element, "text") or not element.text:
                continue
                
            element_text = element.text
            element_size = len(element_text)
            
            if current_size + element_size > 1000 and current_chunk:
                chunks.append(TextChunk(
                    content="".join([e.text for e in current_chunk]),
                    element_ids=[e.id for e in current_chunk],
                    metadata={
                        "element_types": [type(e).__name__ for e in current_chunk]
                    }
                ))
                current_chunk = []
                current_size = 0
            
            if element_size > 1000:
                # Split into smaller chunks
                words = element_text.split()
                sub_chunks = []
                sub_chunk = []
                sub_size = 0
                
                for word in words:
                    word_size = len(word) + 1  # +1 for space
                    if sub_size + word_size > 1000 and sub_chunk:
                        sub_chunks.append(" ".join(sub_chunk))
                        sub_chunk = []
                        sub_size = 0
                    
                    sub_chunk.append(word)
                    sub_size += word_size
                
                if sub_chunk:
                    sub_chunks.append(" ".join(sub_chunk))
                
                # Add each sub-chunk
                for sub_chunk_text in sub_chunks:
                    chunks.append(TextChunk(
                        content=sub_chunk_text,
                        element_ids=[element.id],
                        metadata={
                            "element_types": [type(element).__name__],
                            "is_split": True
                        }
                    ))
            else:
                current_chunk.append(element)
                current_size += element_size
        
        # Add any remaining content
        if current_chunk:
            chunks.append(TextChunk(
                content="".join([e.text for e in current_chunk]),
                element_ids=[e.id for e in current_chunk],
                metadata={
                    "element_types": [type(e).__name__ for e in current_chunk]
                }
            ))
        
        return chunks
    
    async def generate_embeddings(self, elements: List[Dict[str, Any]], document_id: str) -> List[DocumentEmbedding]:
        """Generate embeddings for document elements"""
        embeddings = []
        try:
            logger.info(f"Starting to generate embeddings for {len(elements)} elements")
            
            for i, element in enumerate(elements):
                try:
                    # Get the text content from the element
                    text = element.get("text", "").strip()
                    if not text:
                        logger.debug(f"Skipping element {i} with empty text")
                        continue
                        
                    logger.debug(f"Processing element {i}: {text[:100]}...")
                    
                    # Generate embedding using OpenAI
                    try:
                        embedding = await self.embeddings_model.aembed_query(text)
                        if not embedding:
                            logger.error(f"Failed to generate embedding for element {i}")
                            continue
                            
                        logger.debug(f"Generated embedding for element {i}")
                        
                        # Convert metadata to dictionary if it's an ElementMetadata object
                        metadata = element.get("metadata", {})
                        if hasattr(metadata, "to_dict"):
                            metadata = metadata.to_dict()
                        elif hasattr(metadata, "__dict__"):
                            metadata = metadata.__dict__
                            
                            # Convert any nested objects that have to_dict method
                            for key, value in metadata.items():
                                if hasattr(value, "to_dict"):
                                    metadata[key] = value.to_dict()
                        
                        # Create DocumentEmbedding object
                        doc_embedding = DocumentEmbedding(
                            document_id=document_id,
                            chunk_index=len(embeddings),
                            content=text,
                            embedding=embedding,
                            source_element_id=str(uuid.uuid4()),  # Generate a new UUID if element_id is not present
                            element_type=element.get("type", ""),
                            element_metadata=metadata
                        )
                        
                        embeddings.append(doc_embedding)
                        logger.debug(f"Added embedding {len(embeddings)} to list")
                        
                    except Exception as e:
                        logger.error(f"Error generating embedding for element {i}: {str(e)}")
                        logger.error(f"Element data: {element}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error processing element {i}: {str(e)}")
                    continue
                    
            logger.info(f"Generated {len(embeddings)} embeddings out of {len(elements)} elements")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return [] 