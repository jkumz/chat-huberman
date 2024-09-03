## We want to get all video URLs of a YouTube channel, these will be parsed to get video ID
## This video ID will then be used to scrape transcripts - repeat for all videos in the channel

import json
import logging
import scrapetube
import os, getpass
import time

from components.video_processing_tracker import VideoProcessingTracker
from langchain_community.document_loaders import YoutubeLoader
from yt_dlp import YoutubeDL

# TODO: Add error code handling, retries etc
# TODO: Deploy this as a microservice on the web, polling and updating results daily in cloud

class Scraper():

    '''
    @param channel_username - username of YouTube channel to scrape
    Constructor which takes the channel username as a parameter
    '''
    def __init__(self, channel_username, youtube_loader=YoutubeLoader):
        self.channel_username = channel_username
        self.youtube_loader = youtube_loader

    '''
    -- HELPER METHOD -- HELPER METHOD -- HELPER METHOD -- HELPER METHOD --
    Uses the channel username defined during instantion to put all
    the YT vid URLs from that channel into a list and returns the list
    '''
    def __get_urls(self):
        try:
            videos = scrapetube.get_channel(channel_username=self.channel_username)
            video_urls = [f"https://www.youtube.com/watch?v={video['videoId']}" for video in videos]
            logging.info(f"Count of video URLs: {len(video_urls)}")
            return video_urls

        except json.JSONDecodeError as e:
            logging.debug(f"Error decoding JSON response: {e}")
            return []  # Return an empty list
        except Exception as e:
            logging.debug(f"An error occurred: {e}")
            return []  # Return an empty list


    '''
    @param document - raw transcript string of a youtube video

    -- HELPER METHOD -- HELPER METHOD -- HELPER METHOD -- HELPER METHOD --
    Takes a raw video transcript string and splits into a list of strings of
    max size of 100,000 character to meet demands of AI21 Semantic Splitter.
    This list is returned for processing.
    '''
    def __prepare_document_chunks(self, document):
        # Use list comprehension to split the string into processable chunks
        chunk_size = 99999
        return [document[i:i + chunk_size] for i in range(0, len(document), chunk_size)]

    '''
    @param video_urls - A list of YouTube video URLs
    @param count - Optional count of transcripts to scrape - gets all if left blank
    Returns a list of dictionaries containing information needed to further process the videos for RAG:
    {
        "url" : url of video,
        "chunks" : a list of Strings < 100,000 chars
    }
    '''
    def scrape_and_preprocess(self, count=None):
        video_urls = self.__get_urls()
        chunked_transcripts = [] # Stores dictionaries of { url : [chunks] } format

        languages = ["en-US", "en"] # We only want transcripts in these languages
        for index, url in enumerate(video_urls[:count] if count else video_urls):
            # TODO - Check if already processed - if it is don't need to repeat - use metadata
            # Check if has english transcript
            has_valid_transcript = False
            for lang in languages:
                try:
                    loader = self.youtube_loader.from_youtube_url(url, language=lang)
                    video_transcript = loader.load()
                    has_valid_transcript = True
                    break
                except Exception as exception:
                    logging.debug(f"Attempt with language '{lang}' failed: {exception}")
            if not has_valid_transcript:
                logging.info(f"No suitable transcript found for video {index+1} - Skipping...")
                continue

            # Split the document semantically - semantic splitter can only chunk 99,999 chars at a time
            # so we use List Comprehension to prepare the chunks to be semantically split
            doc_chunks = [
                chunk
                for doc in video_transcript
                for chunk in self.__prepare_document_chunks(doc.page_content)
            ]
            chunked_transcripts.append({"url":url, "chunks":doc_chunks})
            logging.info(f"Saved transcript {index+1}")
        return chunked_transcripts