import logging
import os
import sys

from dotenv import load_dotenv
from transcript_indexer import Indexer
from transcript_scraper import Scraper

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    # Initialize the Scraper with the Huberman Lab channel
    channel_username = "hubermanlab"
    scraper = Scraper(channel_username)

    # Scrape and preprocess transcripts
    logger.info(
        f"Starting to scrape transcripts using channel username: {channel_username}"
    )
    chunked_transcripts = scraper.scrape_and_preprocess()
    logger.info(f"Scraped and preprocessed {len(chunked_transcripts)} videos")

    # Initialize the Indexer with API keys from environment variables
    indexer = Indexer(
        ai21_api_key=os.getenv("AI21_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    # Process and index each video's transcript chunks
    for video in chunked_transcripts:
        url = video["url"]
        chunks = video["chunks"]
        video_id = url.split("v=")[1]  # Extract video ID from URL

        logger.info(f"Processing and indexing video: {url}")
        try:
            indexer.process_and_index_chunks(url, chunks, video_id)
            logger.info(f"Successfully processed and indexed video: {video_id}")
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")

    logger.info("Completed processing and indexing all videos")


if __name__ == "__main__":
    main()
