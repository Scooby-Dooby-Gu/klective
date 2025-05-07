# K - Tesseract

K - Tesseract is a document processing and timeline extraction application that helps users analyze and visualize chronological events from their documents.

## Features

- Document upload and management
- Automatic timeline event extraction
- Interactive timeline visualization
- Document search and filtering
- Collection organization
- Real-time processing status updates

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`
   - Add your Supabase configuration:
     ```
     SUPABASE_URL=your_supabase_project_url
     SUPABASE_KEY=your_supabase_anon_key
     ```

5. Database Setup:
   - Create a new project in Supabase
   - Run the following SQL migrations in your Supabase SQL editor:

   ```sql
   -- Create collections table
   CREATE TABLE collections (
       id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
       name TEXT NOT NULL,
       description TEXT,
       user_id UUID NOT NULL,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Create documents table
   CREATE TABLE documents (
       id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
       collection_id UUID REFERENCES collections(id),
       file_name TEXT NOT NULL,
       file_path TEXT NOT NULL,
       file_size INTEGER NOT NULL,
       mime_type TEXT NOT NULL,
       user_id UUID NOT NULL,
       processed BOOLEAN DEFAULT FALSE,
       processed_at TIMESTAMP WITH TIME ZONE,
       summary TEXT,
       uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Create timeline_events table
   CREATE TABLE timeline_events (
       id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
       document_id UUID REFERENCES documents(id),
       event_date TIMESTAMP WITH TIME ZONE NOT NULL,
       title TEXT NOT NULL,
       description TEXT,
       category TEXT,
       actors TEXT[],
       location TEXT,
       importance INTEGER CHECK (importance BETWEEN 1 AND 10),
       confidence_score FLOAT CHECK (confidence_score BETWEEN 0 AND 1),
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Create api_calls table
   CREATE TABLE api_calls (
       id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
       document_id UUID REFERENCES documents(id),
       api_type TEXT NOT NULL,
       api_calls INTEGER DEFAULT 0,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Create RLS policies
   ALTER TABLE collections ENABLE ROW LEVEL SECURITY;
   ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
   ALTER TABLE timeline_events ENABLE ROW LEVEL SECURITY;
   ALTER TABLE api_calls ENABLE ROW LEVEL SECURITY;

   -- Collections policies
   CREATE POLICY "Users can view their own collections"
       ON collections FOR SELECT
       USING (auth.uid() = user_id);

   CREATE POLICY "Users can insert their own collections"
       ON collections FOR INSERT
       WITH CHECK (auth.uid() = user_id);

   -- Documents policies
   CREATE POLICY "Users can view their own documents"
       ON documents FOR SELECT
       USING (auth.uid() = user_id);

   CREATE POLICY "Users can insert their own documents"
       ON documents FOR INSERT
       WITH CHECK (auth.uid() = user_id);

   -- Timeline events policies
   CREATE POLICY "Users can view their own timeline events"
       ON timeline_events FOR SELECT
       USING (EXISTS (
           SELECT 1 FROM documents
           WHERE documents.id = timeline_events.document_id
           AND documents.user_id = auth.uid()
       ));

   -- API calls policies
   CREATE POLICY "Users can view their own API calls"
       ON api_calls FOR SELECT
       USING (EXISTS (
           SELECT 1 FROM documents
           WHERE documents.id = api_calls.document_id
           AND documents.user_id = auth.uid()
       ));
   ```

## Running the Application

```bash
streamlit run run.py
```

## License

MIT License 
