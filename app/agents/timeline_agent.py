from typing import List, Dict, Any
from uuid import UUID
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

from app.models.pydantic_models import TimelineEvent

class TimelineGeneratorAgent:
    """Agent responsible for timeline visualization"""
    
    def generate_document_timeline(self, events: List[TimelineEvent]) -> go.Figure:
        """Create timeline visualization for a single document"""
        if not events:
            return self._create_empty_timeline()
            
        df = self._prepare_timeline_data(events)
        return self._create_timeline_figure(df, single_document=True)
    
    def generate_collection_timeline(self, events: List[TimelineEvent]) -> go.Figure:
        """Create aggregated timeline for multiple documents"""
        if not events:
            return self._create_empty_timeline()
            
        df = self._prepare_timeline_data(events)
        return self._create_timeline_figure(df, single_document=False)
    
    def _prepare_timeline_data(self, events: List[TimelineEvent]) -> pd.DataFrame:
        """Convert events to DataFrame for visualization"""
        data = []
        
        for event in events:
            data.append({
                "id": str(event.id),
                "date": event.event_date,
                "title": event.title,
                "description": event.description,
                "importance": event.importance,
                "category": event.category or "Uncategorized",
                "actors": ", ".join(event.actors),
                "location": event.location or "Unknown",
                "confidence": event.confidence_score,
                "document_id": str(event.document_id)
            })
        
        return pd.DataFrame(data)
    
    def _create_timeline_figure(self, df: pd.DataFrame, single_document: bool) -> go.Figure:
        """Create Plotly timeline figure"""
        if single_document:
            color_col = "category"
            title = "Document Timeline"
        else:
            color_col = "document_id"
            title = "Collection Timeline"
        
        fig = px.scatter(
            df,
            x="date",
            y="importance",
            color=color_col,
            size="confidence",
            hover_data={
                "date": True,
                "title": True,
                "description": True,
                "actors": True,
                "location": True,
                "importance": True,
                "confidence": True,
                "document_id": True
            },
            title=title
        )
        
        # Customize layout
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
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
                title="Importance",
                showgrid=True,
                zeroline=True,
                showline=True,
                showticklabels=True
            )
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate="<br>".join([
                "<b>%{customdata[1]}</b>",
                "Date: %{customdata[0]}",
                "Description: %{customdata[2]}",
                "Actors: %{customdata[3]}",
                "Location: %{customdata[4]}",
                "Importance: %{customdata[5]}",
                "Confidence: %{customdata[6]}",
                "<extra></extra>"
            ])
        )
        
        return fig
    
    def _create_empty_timeline(self) -> go.Figure:
        """Create an empty timeline figure"""
        fig = go.Figure()
        fig.update_layout(
            height=600,
            title="No events to display",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Importance")
        )
        return fig
    
    def merge_events(self, event_sets: List[List[TimelineEvent]]) -> List[TimelineEvent]:
        """Combine and deduplicate events from multiple documents"""
        all_events = []
        seen_events = set()
        
        for events in event_sets:
            for event in events:
                # Create a unique identifier for the event
                event_key = (
                    event.event_date.isoformat(),
                    event.title,
                    event.description
                )
                
                if event_key not in seen_events:
                    seen_events.add(event_key)
                    all_events.append(event)
        
        # Sort events by date
        return sorted(all_events, key=lambda x: x.event_date) 