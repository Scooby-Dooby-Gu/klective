import streamlit as st
import os
import asyncio
from datetime import datetime
from typing import List, Optional
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from uuid import UUID, uuid5, NAMESPACE_DNS
import logging
import requests
import tempfile
import plotly.graph_objects as go
import threading

from app.utils.supabase import SupabaseClient
from app.agents.processor_agent import DocumentProcessorAgent
from app.agents.timeline_extractor_agent import TimelineExtractorAgent
from app.models.pydantic_models import Collection, Document, TimelineEvent
from app.utils.logger import setup_logger

# Set up logging
logger = setup_logger()

# Load environment variables
load_dotenv()

# Initialize clients
supabase = SupabaseClient()

# Initialize Document Processor Agent with OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
processor_agent = DocumentProcessorAgent(openai_api_key=openai_api_key)

# Initialize Timeline Extractor Agent
timeline_extractor = TimelineExtractorAgent(openai_api_key=openai_api_key)

# For development/testing, use a deterministic UUID for the test user
USER_ID = uuid5(NAMESPACE_DNS, "test_user@example.com")

# Initialize session state for timeline processor
if 'timeline_processor_running' not in st.session_state:
    st.session_state.timeline_processor_running = False

async def start_timeline_processor():
    """Start the timeline processor in the background."""
    try:
        # Start the processor in the background
        asyncio.create_task(timeline_extractor.start_background_processing())
        
        # Update session state
        st.session_state.timeline_processor_running = True
        
        logger.info("Timeline processor started successfully")
    except Exception as e:
        logger.error(f"Error starting timeline processor: {str(e)}")
        st.session_state.timeline_processor_running = False
        raise

async def stop_timeline_processor():
    """Stop the timeline processor background task."""
    if st.session_state.timeline_processor_running:
        st.session_state.timeline_processor_running = False
        await timeline_extractor.stop_background_processing()

async def process_document_async(doc_id: str):
    """Async wrapper for document processing."""
    try:
        logger.debug(f"Starting to process document {doc_id}")
        st.info(f"Starting to process document {doc_id}...")
        
        # Process the document using the document ID
        logger.debug("Starting document processing with agent")
        st.info("Starting document processing with agent...")
        result = await processor_agent.process_document(doc_id)
        
        if result:
            logger.debug(f"Document processing completed successfully. Events: {len(result.events)}, Embeddings: {len(result.embeddings)}")
            st.info("Document processing completed successfully")
            # Log the processing result for debugging
            st.json({
                "document_id": str(result.document_id),
                "summary": result.summary,
                "events_count": len(result.events),
                "embeddings_count": len(result.embeddings)
            })
            
            # Save the processing result
            logger.debug("Saving processing result to database")
            st.info("Saving processing result to database...")
            await supabase.save_processing_result(doc_id, result)
            logger.debug("Processing result saved successfully")
            st.success("Processing result saved successfully")
            return True
        else:
            error_msg = "Document processing returned no result"
            logger.error(error_msg)
            st.error(error_msg)
            return False
            
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main entry point for the Streamlit application."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Klective",
        page_icon="app/assets/gradient.svg",
        layout="wide"
    )

    # Initialize session states
    if 'show_logs' not in st.session_state:
        st.session_state.show_logs = True
    if 'timeline_processor_running' not in st.session_state:
        st.session_state.timeline_processor_running = False

    # Initialize error container at the top of the page
    error_container = st.container()
    
    # Add a toggle for log visibility in the sidebar
    col1, col2 = st.sidebar.columns([1, 4])
    with col1:
        st.image("app/assets/black.svg", width=30)
    with col2:
        st.markdown("<h1 style='margin: 0;'>Tesseract</h1>", unsafe_allow_html=True)
    st.sidebar.checkbox("Show Logs", value=st.session_state.show_logs, key="show_logs")
    
    # Start timeline processor if not already running
    if not st.session_state.timeline_processor_running:
        logger.info("Starting timeline processor...")
        try:
            # Create a new event loop for the timeline processor
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_timeline_processor())
            st.session_state.timeline_processor_running = True
        except Exception as e:
            logger.error(f"Error starting timeline processor: {str(e)}")
            st.error(f"Failed to start timeline processor: {str(e)}")
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Collections", "Documents", "Timeline", "Search"]
    )

    # Helper function for logging
    def log_message(message: str, level: str = "info"):
        """Log a message with the specified level."""
        if st.session_state.show_logs:
            if level == "error":
                logger.error(message)
                st.error(message, icon="🚨")
            elif level == "warning":
                logger.warning(message)
                st.warning(message, icon="⚠️")
            elif level == "success":
                logger.info(message)
                st.success(message, icon="✅")
            else:
                logger.info(message)
                st.info(message, icon="ℹ️")

    # Helper functions
    def get_collections() -> List[Collection]:
        """Fetch collections for the current user."""
        return supabase.get_collections(USER_ID)

    def get_documents(collection_id: Optional[str] = None) -> List[Document]:
        """Fetch documents for the current user, optionally filtered by collection."""
        return supabase.get_documents(collection_id, USER_ID)

    def process_document(document: Document) -> None:
        """Process a document and save results."""
        try:
            log_message(f"Starting to process document: {document.file_name}")
            
            # Create an async task to process the document
            async def process():
                result = await processor_agent.process_document(document.id)
                if result:
                    await supabase.save_processing_result(document.id, result)
                    log_message(f"Document processed successfully: {document.file_name}", "success")
                else:
                    log_message("Document processing returned no result", "error")
            
            # Run the async task
            asyncio.run(process())
            
        except Exception as e:
            with error_container:
                log_message(f"Error processing document: {str(e)}", "error")
                log_message(f"Error type: {type(e).__name__}", "error")
                import traceback
                log_message(f"Traceback: {traceback.format_exc()}", "error")

    # Dashboard page
    if page == "Dashboard":
        st.title("Dashboard")
        
        # Fetch collections and documents
        collections = get_collections()
        documents = get_documents()
        
        # Get timeline events
        events_response = supabase.events.select("*").execute()
        total_events = len(events_response.data) if events_response.data else 0
        
        # Get API call counts
        api_calls = supabase.client.table("api_calls").select("api_type", "api_calls").execute()
        total_document_api_calls = sum(call["api_calls"] for call in api_calls.data if call["api_type"] == "document")
        total_timeline_api_calls = sum(call["api_calls"] for call in api_calls.data if call["api_type"] == "timeline")
        
        # Group events by document
        doc_events = {}
        if events_response.data:
            for event in events_response.data:
                doc_id = event['document_id']
                if doc_id not in doc_events:
                    doc_events[doc_id] = []
                doc_events[doc_id].append(event)
        
        # Calculate metrics
        total_collections = len(collections)
        total_documents = len(documents)
        processed_documents = sum(1 for doc in documents if doc.processed)
        documents_with_events = len(doc_events)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Collections", total_collections)
        with col2:
            st.metric("Total Documents", total_documents)
        with col3:
            st.metric("Processed Documents", processed_documents)
        with col4:
            st.metric("Timeline Events", total_events)
        
        # Display API call metrics
        st.subheader("API Usage")
        api_col1, api_col2 = st.columns(2)
        with api_col1:
            st.metric("Document Processing API Calls", total_document_api_calls)
        with api_col2:
            st.metric("Timeline Processing API Calls", total_timeline_api_calls)
        
        # Show timeline processing status
        st.subheader("Timeline Processing Status")
        if st.session_state.timeline_processor_running:
            st.success("Timeline processor is running and checking for new documents every 5 minutes")
        else:
            st.warning("Timeline processor is not running")
        
        # Show documents with events and their API usage
        if documents_with_events > 0:
            st.subheader("Documents with Timeline Events")
            for doc_id, events in doc_events.items():
                doc = next((d for d in documents if str(d.id) == doc_id), None)
                if doc:
                    # Get API calls for this document
                    doc_api_calls = supabase.client.table("api_calls").select("*").eq("document_id", doc_id).execute()
                    doc_calls = sum(call["api_calls"] for call in doc_api_calls.data if call["api_type"] == "document")
                    timeline_calls = sum(call["api_calls"] for call in doc_api_calls.data if call["api_type"] == "timeline")
                    
                    st.write(f"- {doc.file_name}:")
                    st.write(f"  • Events: {len(events)}")
                    st.write(f"  • Document Processing API Calls: {doc_calls}")
                    st.write(f"  • Timeline Processing API Calls: {timeline_calls}")
        
        # Recent activity
        st.subheader("Recent Activity")
        recent_docs = sorted(documents, key=lambda x: x.uploaded_at, reverse=True)[:5]
        for doc in recent_docs:
            st.write(f"- {doc.file_name} ({doc.uploaded_at.strftime('%Y-%m-%d %H:%M')})")

    # Collections page
    elif page == "Collections":
        st.title("Collections")
        
        # Create new collection
        with st.form("new_collection"):
            st.subheader("Create New Collection")
            name = st.text_input("Collection Name")
            description = st.text_area("Description")
            if st.form_submit_button("Create"):
                if name:
                    try:
                        collection = supabase.create_collection(name, description, USER_ID)
                        st.success(f"Collection created: {name}")
                    except Exception as e:
                        with error_container:
                            st.error(f"Error creating collection: {str(e)}", icon="🚨")
                else:
                    with error_container:
                        st.error("Collection name is required", icon="🚨")
        
        # List collections
        st.subheader("Your Collections")
        collections = get_collections()
        for collection in collections:
            with st.expander(f"{collection.name} ({len(get_documents(collection.id))} documents)"):
                st.write(collection.description)
                if st.button("View Documents", key=f"view_{collection.id}"):
                    st.session_state.current_collection = collection.id
                    st.rerun()

    # Documents page
    elif page == "Documents":
        st.title("Documents")
        
        # Document upload section
        st.subheader("Upload New Document")
        with st.form("upload_document"):
            # Collection selection
            collections = supabase.get_collections(user_id=UUID("f5889c5c-28b0-56eb-9145-e3b9457490d3"))
            collection_options = {str(col.id): col.name for col in collections}
            selected_collection = st.selectbox(
                "Select Collection",
                options=list(collection_options.keys()),
                format_func=lambda x: collection_options[x]
            )
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=['pdf', 'doc', 'docx', 'txt']
            )
            
            if st.form_submit_button("Upload"):
                if uploaded_file and selected_collection:
                    try:
                        # Save the file temporarily
                        file_path = f"temp_{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Get file name and mime type
                        file_name = uploaded_file.name
                        mime_type = uploaded_file.type
                        
                        # Upload to Supabase
                        doc = supabase.upload_document(
                            file_path=file_path,
                            file_name=file_name,
                            mime_type=mime_type,
                            collection_id=selected_collection,
                            user_id=UUID("f5889c5c-28b0-56eb-9145-e3b9457490d3")
                        )
                        
                        # Clean up temp file
                        os.remove(file_path)
                        
                        if doc:
                            st.success(f"Document uploaded successfully: {uploaded_file.name}")
                            st.rerun()  # Refresh the page to show the new document
                        else:
                            st.error("Failed to upload document")
                    except Exception as e:
                        st.error(f"Error uploading document: {str(e)}")
                else:
                    st.error("Please select a collection and upload a file")
        
        # Get all collections and documents
        try:
            logger.debug("Fetching collections and documents")
            log_message("Fetching collections and documents...")
            collections = supabase.get_collections(user_id=UUID("f5889c5c-28b0-56eb-9145-e3b9457490d3"))
            documents = supabase.get_documents(user_id=UUID("f5889c5c-28b0-56eb-9145-e3b9457490d3"))
            logger.debug(f"Found {len(collections)} collections and {len(documents)} documents")
            log_message(f"Found {len(collections)} collections and {len(documents)} documents", "success")
        except Exception as e:
            logger.error("Error fetching collections or documents", exc_info=True)
            with error_container:
                log_message(f"Error fetching collections or documents: {str(e)}", "error")
                log_message(f"Error type: {type(e).__name__}", "error")
                import traceback
                log_message(f"Traceback: {traceback.format_exc()}", "error")
            collections = []
            documents = []
        
        # Create a mapping of collection IDs to names
        collection_names = {str(col.id): col.name for col in collections}
        
        # Filter unprocessed documents
        unprocessed_docs = [doc for doc in documents if not doc.processed]
        
        if unprocessed_docs:
            st.subheader("Unprocessed Documents")
            for doc in unprocessed_docs:
                with st.expander(f"{doc.file_name} - {doc.mime_type}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"Collection: {collection_names.get(str(doc.collection_id), 'Unknown Collection')}")
                        st.write(f"Uploaded: {doc.uploaded_at}")
                        st.write(f"Size: {doc.file_size} bytes")
                        st.write(f"Document ID: {doc.id}")
                        st.write(f"File Path: {doc.file_path}")
                    with col2:
                        if st.button("Delete", key=f"delete_unprocessed_{doc.id}"):
                            try:
                                if supabase.delete_document(doc.id, UUID("f5889c5c-28b0-56eb-9145-e3b9457490d3")):
                                    log_message(f"Deleted document: {doc.file_name}", "success")
                                    st.rerun()  # Refresh the page to update the list
                                else:
                                    log_message("Failed to delete document", "error")
                            except Exception as e:
                                log_message(f"Error deleting document: {str(e)}", "error")
                    
                    if st.button("Process Document", key=f"process_{doc.id}"):
                        try:
                            log_message(f"Starting processing for document: {doc.file_name}")
                            asyncio.run(process_document_async(doc.id))
                            log_message(f"Processing started for {doc.file_name}", "success")
                            st.rerun()  # Refresh the page to update the list
                        except Exception as e:
                            log_message(f"Error processing document: {str(e)}", "error")
                            log_message(f"Error type: {type(e).__name__}", "error")
                            import traceback
                            log_message(f"Traceback: {traceback.format_exc()}", "error")
        
        # Display processed documents
        processed_docs = [doc for doc in documents if doc.processed]
        if processed_docs:
            st.subheader("Processed Documents")
            for doc in processed_docs:
                with st.expander(f"{doc.file_name} - {doc.mime_type}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"Collection: {collection_names.get(str(doc.collection_id), 'Unknown Collection')}")
                        st.write(f"Uploaded: {doc.uploaded_at}")
                        st.write(f"Processed: {doc.processed_at}")
                        st.write(f"Size: {doc.file_size} bytes")
                        st.write(f"Summary: {doc.summary or 'No summary available'}")
                    with col2:
                        if st.button("Delete", key=f"delete_processed_{doc.id}"):
                            try:
                                if supabase.delete_document(doc.id, UUID("f5889c5c-28b0-56eb-9145-e3b9457490d3")):
                                    log_message(f"Deleted document: {doc.file_name}", "success")
                                    st.rerun()  # Refresh the page to update the list
                                else:
                                    log_message("Failed to delete document", "error")
                            except Exception as e:
                                log_message(f"Error deleting document: {str(e)}", "error")

    # Timeline page
    elif page == "Timeline":
        st.title("Timeline")
        
        # Select collection or all documents
        collections = get_collections()
        collection_options = ["All Documents"] + [c.id for c in collections]
        selected_collection = st.selectbox(
            "Select Collection",
            options=collection_options,
            format_func=lambda x: "All Documents" if x == "All Documents" else next(c.name for c in collections if c.id == x)
        )
        
        # Get events
        if selected_collection == "All Documents":
            documents = get_documents()
        else:
            documents = get_documents(selected_collection)
        
        events = []
        for doc in documents:
            doc_events = supabase.get_events(doc.id)
            events.extend(doc_events)
        
        if events:
            # Convert events to DataFrame with date validation
            valid_events = []
            for event in events:
                try:
                    # Convert to datetime and validate
                    event_date = pd.to_datetime(event.event_date)
                    # Convert to timezone-naive datetime if it's timezone-aware
                    if event_date.tz is not None:
                        event_date = event_date.tz_localize(None)
                    
                    if pd.Timestamp.min <= event_date <= pd.Timestamp.max:
                        valid_events.append({
                            'date': event_date,
                            'title': event.title,
                            'description': event.description,
                            'importance': event.importance,
                            'actors': ", ".join(event.actors) if event.actors else "Unknown",
                            'location': event.location or "Unknown",
                            'confidence': event.confidence_score
                        })
                    else:
                        st.warning(f"Skipping event with out-of-bounds date: {event.title}")
                except Exception as e:
                    st.warning(f"Skipping event with invalid date: {event.title} - {str(e)}")
                    continue
            
            if not valid_events:
                st.warning("No valid events found with proper dates")
                return
            
            df = pd.DataFrame(valid_events)
            
            # Filter by date range if selected
            date_range = st.date_input(
                "Date Range",
                value=(
                    df['date'].min().date(),
                    df['date'].max().date()
                )
            )
            
            if date_range:
                # Convert date_range to timezone-naive datetime
                start_date = pd.Timestamp(date_range[0]).tz_localize(None)
                end_date = pd.Timestamp(date_range[1]).tz_localize(None)
                
                df = df[
                    (df['date'].dt.date >= start_date.date()) &
                    (df['date'].dt.date <= end_date.date())
                ]
            
            # Create timeline visualization
            fig = go.Figure()
            
            # Add a line connecting all events
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=[0] * len(df),  # All points on the same y-level
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
            
            # Add event nodes
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=[0] * len(df),  # All points on the same y-level
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['importance'],  # Color by importance
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Importance')
                ),
                text=df['title'],  # Show title on hover
                hoverinfo='text',
                customdata=df[[
                    'description',
                    'importance',
                    'actors',
                    'location',
                    'confidence'
                ]].values,
                hovertemplate="<br>".join([
                    "<b>%{text}</b>",
                    "Date: %{x}",
                    "Description: %{customdata[0]}",
                    "Importance: %{customdata[1]}",
                    "Actors: %{customdata[2]}",
                    "Location: %{customdata[3]}",
                    "Confidence: %{customdata[4]}",
                    "<extra></extra>"
                ])
            ))
            
            # Customize layout
            fig.update_layout(
                height=400,  # Reduced height since we're using a single line
                showlegend=False,
                hovermode="closest",
                xaxis=dict(
                    title="Date",
                    showgrid=True,
                    zeroline=True,
                    showline=True,
                    showticklabels=True,
                    rangeslider=dict(visible=True)
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    range=[-0.5, 0.5]  # Constrain y-axis to show only the timeline
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add event filtering options
            st.subheader("Filter Events")
            col1, col2 = st.columns(2)
            
            with col1:
                min_importance = st.slider(
                    "Minimum Importance",
                    min_value=1,
                    max_value=10,
                    value=1
                )
                
                # Get unique actors from all events
                all_actors = set()
                for actors_str in df['actors']:
                    if actors_str and actors_str != "Unknown":
                        all_actors.update([actor.strip() for actor in actors_str.split(",")])
                all_actors = sorted(list(all_actors))
                
                selected_actors = st.multiselect(
                    "Actors",
                    options=all_actors,
                    default=all_actors
                )
            
            with col2:
                min_confidence = st.slider(
                    "Minimum Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
            
            # Apply filters
            filtered_df = df[
                (df['importance'] >= min_importance) &
                (df['confidence'] >= min_confidence)
            ]
            
            # Filter by actors if any are selected
            if selected_actors:
                filtered_df = filtered_df[
                    filtered_df['actors'].apply(
                        lambda x: any(actor in x for actor in selected_actors)
                    )
                ]
            
            # Display filtered events in a table
            st.subheader("Filtered Events")
            st.dataframe(
                filtered_df[[
                    'date', 'title', 'description', 'importance',
                    'actors', 'location', 'confidence'
                ]],
                use_container_width=True
            )
            
        else:
            st.info("No events found in selected documents")

    # Search page
    elif page == "Search":
        st.title("Search Documents")
        
        # Search input
        query = st.text_input("Enter your search query")
        if query:
            # Perform search
            results = supabase.search_documents(query, USER_ID)
            
            if results:
                st.subheader("Search Results")
                for doc in results:
                    with st.expander(f"{doc.file_name} (Score: {doc.similarity:.2f})"):
                        st.write("Summary:", doc.summary)
                        if st.button("View Timeline", key=f"timeline_{doc.id}"):
                            events = supabase.get_events(doc.id)
                            if events:
                                df = pd.DataFrame([{
                                    'date': pd.to_datetime(event.event_date),
                                    'title': event.title,
                                    'description': event.description,
                                    'importance': event.importance,
                                    'category': event.category
                                } for event in events])
                                
                                fig = px.timeline(
                                    df,
                                    x_start="date",
                                    x_end="date",
                                    y="title",
                                    color="category",
                                    hover_data=["description", "importance"],
                                    title=f"Timeline for {doc.file_name}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No events found for this document")
            else:
                st.info("No documents found matching your query")

if __name__ == "__main__":
    main() 