# Processes document chunks, embeds and indexes in pinecone
import os
import logging
from typing import List
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_openai import OpenAIEmbeddings
from video_processing_tracker import VideoProcessingTracker
from yt_dlp import YoutubeDL
from pinecone import Pinecone

logger = logging.getLogger(__name__)

class Indexer:
    openai_embedding_model = "text-embedding-3-large"

    """
    Constructor for the Indexer class.

    Parameters:
    - ai21_api_key (str): API key for AI21.
    - openai_api_key (str): API key for OpenAI.
    - pinecone_api_key (str): API key for Pinecone.
    """

    def __init__(self, ai21_api_key, openai_api_key, pinecone_api_key):
        self.ai21_api_key = ai21_api_key
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key

        self.tracker = VideoProcessingTracker()

        self.semantic_text_splitter = AI21SemanticTextSplitter(
            api_key=self.ai21_api_key
        )
        self.embedding_model = OpenAIEmbeddings(
            api_key=self.openai_api_key, model=self.openai_embedding_model
        )
        self.pinecone = Pinecone(
            api_key=self.pinecone_api_key, source_tag=os.environ["INDEX_SOURCE_TAG"]
        )

        self.index = self.pinecone.Index(
            host=os.environ["INDEX_HOST"], name=os.environ["INDEX_NAME"]
        )

    def __extract_metadata(self, video_url):
        opts = {}
        try:
            with YoutubeDL(opts) as yt:
                info = yt.extract_info(video_url, download=False)
                if len(info) == 0:
                    return {}
                data = {"url": video_url, "title": info.get("title")}
                return data
        except Exception as e:
            logger.debug(f"Encountered error while extracting metadata: {e}")

    def process_and_index_chunks(
        self, url, doc_chunks: List[str], video_id: str, batch_size: int = 100
    ):
        # Use attributes initialized in the constructor
        title = self.__extract_metadata(video_url=url)["title"]
        logger.info(f"Starting processing for video: {video_id}")
        self.tracker.start_processing(video_id)

        try:
            vectors_to_upsert = []

            for chunk_i, doc_chunk in enumerate(doc_chunks):
                if len(doc_chunk) < 30:
                    continue

                semantic_splits = self.semantic_text_splitter.split_text(doc_chunk)
                for split_i, split in enumerate(semantic_splits):
                    embedding = self.embedding_model.embed_query(split)
                    metadata = {
                        "video_id": video_id,
                        "video_url": url,
                        "video_title": title,
                        "chunk_index": chunk_i,
                        "split_index": split_i,
                        "text": split,
                    }

                    vector_id = f"{video_id}_chunk{chunk_i}_split{split_i}"
                    vectors_to_upsert.append((vector_id, embedding, metadata))

                    if len(vectors_to_upsert) >= batch_size:
                        self.index.upsert(vectors=vectors_to_upsert)
                        vectors_to_upsert = []

            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert)

            self.tracker.complete_processing(video_id)
            logger.info(f"Successfully processed and indexed video: {video_id}")

        except Exception as e:
            logger.debug(f"Error processing video {video_id}: {str(e)}")
            self.tracker.fail_processing(video_id)
            raise e
