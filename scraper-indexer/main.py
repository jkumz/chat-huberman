"""
This is the main entry point for the scraper-indexer application.
It is responsible for initializing the scraper and indexer, and for processing and indexing the transcripts.
These transcripts are then retrieved and used by the chat service.
"""

import os
import schedule
import time

from dotenv import load_dotenv, find_dotenv
from components.transcript_indexer import Indexer
from components.transcript_scraper import Scraper

from components.logger import logger

# Load environment variables
load_dotenv(find_dotenv(filename=".scraper-indexer.env"))

def main():
    # Initialize the Scraper with the Huberman Lab channel
    channel_username = "hubermanlab"
    scraper = Scraper(channel_username)

    # Scrape and preprocess transcripts
    logger.info(
        f"Starting to scrape transcripts using channel username: {channel_username}"
    )
    video_data_with_chunks = scraper.scrape_and_preprocess()
    logger.info(f"Scraped and preprocessed {len(video_data_with_chunks)} videos")

    # Initialize the Indexer with API keys from environment variables
    indexer = Indexer(
        ai21_api_key=os.getenv("AI21_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    # Process and index each video's transcript chunks
    for video_data in video_data_with_chunks:
        url = video_data["url"]
        title = video_data["title"]
        chunks = video_data["chunks"]
        video_id = video_data["id"]

        logger.info(f"Processing and indexing video: {title} - {url}")
        try:
            indexer.process_and_index_chunks(url, chunks, video_id, title)
            logger.info(f"Successfully processed and indexed video: {video_id}")
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")

    logger.info("Completed processing and indexing all videos")

if __name__ == "__main__":
    logger.info("Starting scraper-indexer")
    main()
    logger.info("Completed scraper-indexer")