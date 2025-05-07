import asyncio
import os
from dotenv import load_dotenv
from app.agents.timeline_extractor_agent import TimelineExtractorAgent
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Initialize and start the timeline extractor
    timeline_extractor = TimelineExtractorAgent(openai_api_key=openai_api_key)
    
    try:
        logger.info("Starting timeline extraction background task...")
        await timeline_extractor.start_background_processing()
        
        # Keep the script running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    finally:
        await timeline_extractor.stop_background_processing()
        logger.info("Timeline extraction background task stopped")

if __name__ == "__main__":
    asyncio.run(main()) 