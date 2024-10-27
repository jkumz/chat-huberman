# Processes document chunks, embeds and indexes in pinecone
import os
from typing import List
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_openai import OpenAIEmbeddings
from components.video_processing_tracker import VideoProcessingTracker
from pinecone import Pinecone

from components.logger import logger

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

    def process_and_index_chunks(
        self, url, doc_chunks: List[str], video_id: str, title: str, batch_size: int = 100
    ):
        # Use attributes initialized in the constructor
        logger.info(f"Starting processing for video: {video_id}")
        logger.info(f"Doc chunks: {doc_chunks}")
        # Check if the video is already in "processing" state
        current_status = self.tracker.get_status(video_id)
        if current_status == "processing" or current_status == "failed":
            logger.warning(f"Video {video_id} was already in 'processing' state or 'failed' state. Attempting to delete existing records.")
            try:
                # Generate vector IDs based on the video ID
                vector_ids = [f"{video_id}_chunk{i}_split{j}" 
                          for i in range(len(doc_chunks)) 
                          for j in range(100)] 
                self.index.delete(ids=vector_ids, delete_all=False)
                logger.info(f"Successfully deleted existing records for video {video_id}")
            except Exception as delete_error:
                logger.error(f"Error deleting existing records for video {video_id}: {str(delete_error)}")
                raise
        self.tracker.start_processing(video_id)

        vectors_to_upsert = []

        try:
            upserted = False
            for chunk_i, doc_chunk in enumerate(doc_chunks):
                if len(doc_chunk) < 30:
                    continue

                semantic_splits = self.semantic_text_splitter.split_text(doc_chunk)
                if len(semantic_splits) == 0:
                    logger.debug(f"No semantic splits for chunk {chunk_i} of video {video_id}")
                    continue
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
                        logger.info(f"Upserting {len(vectors_to_upsert)} vectors for video {video_id}")
                        self.index.upsert(vectors=vectors_to_upsert)
                        upserted = True
                        vectors_to_upsert = []

            if vectors_to_upsert:
                logger.info(f"Upserting {len(vectors_to_upsert)} vectors for video {video_id}")
                self.index.upsert(vectors=vectors_to_upsert)
                upserted = True

            if not upserted:
                logger.warning(f"No vectors upserted for video {video_id}")
            else:
                self.tracker.complete_processing(video_id)
                logger.info(f"Successfully processed and indexed video: {video_id}")

        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")
            raise e
        except KeyboardInterrupt:
            logger.warning(f"Processing interrupted for video {video_id}")
            raise
        finally:
            if self.tracker.get_status(video_id) != "completed":
                logger.warning(f"Processing incomplete for video {video_id}. Cleaning up...")
                self.tracker.fail_processing(video_id)
                # Delete all vectors in the index which have the video id in the metadata
                try:
                    # Generate vector IDs based on the video ID
                    vector_ids = [f"{video_id}_chunk{i}_split{j}" 
                                for i in range(len(doc_chunks)) 
                                for j in range(100)] 
                    self.index.delete(ids=vector_ids, delete_all=False)
                    logger.info(f"Deleted incomplete vectors for video {video_id}")
                except Exception as delete_error:
                    logger.error(f"Error deleting vectors for video {video_id}: {str(delete_error)}")
