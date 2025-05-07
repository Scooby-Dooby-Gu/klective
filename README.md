# Klective

Klective is a document processing and timeline extraction application that helps users analyze and visualize chronological events from their documents.

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

## Running the Application

```bash
streamlit run run.py
```

## License

MIT License 