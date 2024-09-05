## We want to get all video URLs of a YouTube channel, these will be parsed to get video ID
## This video ID will then be used to scrape transcripts - repeat for all videos in the channel
# TODO: Add error code handling, retries etc
# TODO: Deploy this as a microservice on the web, polling and updating results daily in cloud

import json
import logging
import scrapetube
from langchain_community.document_loaders import YoutubeLoader

logger = logging.getLogger(__name__)


class Scraper:
    """
    @param channel_username - username of YouTube channel to scrape
    Constructor which takes the channel username as a parameter
    """

    def __init__(self, channel_username, youtube_loader=YoutubeLoader):
        self.channel_username = channel_username
        self.youtube_loader = youtube_loader

    """
    -- HELPER METHOD -- HELPER METHOD -- HELPER METHOD -- HELPER METHOD --
    Uses the channel username defined during instantion to put all
    the YT vid URLs from that channel into a list and returns the list
    """

    def __get_all_video_data(self):
        try:
            videos = scrapetube.get_channel(channel_username=self.channel_username)
            all_video_data = []
            for video in videos:
                video_url = f"https://www.youtube.com/watch?v={video['videoId']}"
                title = video['title']['runs'][0]['text']
                video_id = video["videoId"]
                all_video_data.append(
                    {
                        "video_id": video_id,
                        "video_url": video_url,
                        "title": title,
                    }
                )
            logger.info(f"Count of video URLs: {len(all_video_data)}")
            return all_video_data

        except json.JSONDecodeError as e:
            logger.debug(f"Error decoding JSON response: {e}")
            return []  # Return an empty list
        except Exception as e:
            logger.debug(f"An error occurred: {e}")
            return []  # Return an empty list

    """
    @param document - raw transcript string of a youtube video

    -- HELPER METHOD -- HELPER METHOD -- HELPER METHOD -- HELPER METHOD --
    Takes a raw video transcript string and splits into a list of strings of
    max size of 100,000 character to meet demands of AI21 Semantic Splitter.
    This list is returned for processing.
    """

    def __prepare_document_chunks(self, document):
        # Use list comprehension to split the string into processable chunks
        chunk_size = 99999
        return [
            document[i : i + chunk_size] for i in range(0, len(document), chunk_size)
        ]

    """
    @param video_urls - A list of YouTube video URLs
    @param count - Optional count of transcripts to scrape - gets all if left blank
    Returns a list of dictionaries containing information needed to further process the videos for RAG:
    {
        "url" : url of video,
        "chunks" : a list of Strings < 100,000 chars
    }
    """

    def scrape_and_preprocess(self, count=None):
        all_video_data = self.__get_all_video_data()
        video_data_with_chunks = []  # Stores dictionaries with video data and chunks

        languages = ["en-US", "en"]  # We only want transcripts in these languages
        for video_data in all_video_data[:count] if count else all_video_data:
            has_valid_transcript = False
            for lang in languages:
                try:
                    loader = self.youtube_loader.from_youtube_url(video_data["video_url"], language=lang)
                    video_transcript = loader.load()
                    has_valid_transcript = True
                    break
                except Exception as exception:
                    logger.debug(f"Attempt with language '{lang}' failed: {exception}")
            if not has_valid_transcript:
                logger.info(f"No suitable transcript found for video: {video_data['title']} - {video_data['video_url']}. Skipping...")
                continue

            # Split the document semantically
            # Semantic splitter can only chunk 99,999 chars at a time
            doc_chunks = [
                chunk
                for doc in video_transcript
                for chunk in self.__prepare_document_chunks(doc.page_content)
            ]
            video_data_with_chunks.append({
                "url": video_data["video_url"],
                "title": video_data["title"],
                "id": video_data["video_id"],
                "chunks": doc_chunks
            })
            logger.info(f"Saved transcript for video: {video_data['title']} - {video_data['video_url']}")
        return video_data_with_chunks
