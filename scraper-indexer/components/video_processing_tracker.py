import os
import sqlite3
from datetime import datetime

from components.logger import logger


class VideoProcessingTracker:
    # Find the video_processing.db file in the /persistence folder
    def find_db_path(self, max_depth=10):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for _ in range(max_depth):
            db_path = os.path.join(current_dir, "persistence", "video_processing.db")
            if os.path.exists(db_path):
                return db_path
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached the root directory
                break
            current_dir = parent_dir
        raise FileNotFoundError(
            "Could not find the video_processing.db file in the persistence directory"
        )

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = self.find_db_path()

        logger.info(f"Initializing VideoProcessingTracker with db_path: {db_path}")
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    # Create table for tracking video processing status
    def create_table(self):
        logger.debug("Creating video_processing_status table if not exists")
        cursor = self.conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS video_processing_status
        (video_id TEXT PRIMARY KEY, 
         status TEXT, 
         start_time TIMESTAMP,
         end_time TIMESTAMP)
        """
        )
        self.conn.commit()

    # Start processing video - insert relevant rows
    def start_processing(self, video_id):
        logger.info(f"Checking status for video_id: {video_id}")
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT status FROM video_processing_status WHERE video_id = ?",
            (video_id,)
        )
        result = cursor.fetchone()
        if result is None or result[0] != "completed":
            logger.info(f"Starting processing for video_id: {video_id}")
            cursor.execute(
                """
                INSERT OR REPLACE INTO video_processing_status
                (video_id, status, start_time) VALUES (?, ?, ?)
                """,
                (video_id, "processing", datetime.now()),
            )
            logger.info(f"Inserted start processing for video_id: {video_id}")
            self.conn.commit()
        else:
            logger.info(f"Skipping processing for video_id: {video_id} as it is already completed")

    # Update relevant rows upon completion
    def complete_processing(self, video_id):
        logger.info(f"Completing processing for video_id: {video_id}")
        cursor = self.conn.cursor()
        cursor.execute(
            """
        UPDATE video_processing_status
        SET status = 'completed', end_time = ?
        WHERE video_id = ?
        """,
            (datetime.now(), video_id),
        )
        self.conn.commit()

    # Mark as failed so that we can restart if failed
    def fail_processing(self, video_id):
        logger.warning(f"Marking processing as failed for video_id: {video_id}")
        cursor = self.conn.cursor()
        cursor.execute(
            """
        UPDATE video_processing_status
        SET status = 'failed', end_time = ?
        WHERE video_id = ?
        """,
            (datetime.now(), video_id),
        )
        self.conn.commit()

    def get_status(self, video_id):
        logger.debug(f"Getting status for video_id: {video_id}")
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT status FROM video_processing_status WHERE video_id = ?", (video_id,)
        )
        result = cursor.fetchone()
        logger.debug(f"Status for video_id {video_id}: {result[0] if result else None}")
        return result[0] if result else None

    def get_unprocessed_videos(self):
        logger.info("Getting unprocessed videos")
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT video_id FROM video_processing_status WHERE status != "completed"'
        )
        result = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(result)} unprocessed videos")
        return result

    def check_if_video_exists_and_completed(self, video_id):
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT video_id, status FROM video_processing_status WHERE video_id = ?',
            (video_id,)
        )
        result = cursor.fetchone()
        exists_and_completed = result is not None and result[1] == "completed"
        if not exists_and_completed:
            if result is None:
                logger.debug(f"Video_id {video_id} does not exist")
            elif result[1] != "completed":
                logger.debug(f"Video_id {video_id} exists but status is not completed")

        return exists_and_completed

    def close(self):
        logger.info("Closing database connection")
        self.conn.close()
