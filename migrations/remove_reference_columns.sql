-- Remove reference columns from timeline_events table
ALTER TABLE timeline_events
    DROP COLUMN IF EXISTS event_references,
    DROP COLUMN IF EXISTS page_or_section_references,
    DROP COLUMN IF EXISTS source_reference; 