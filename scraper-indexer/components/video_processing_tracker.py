import os
import psycopg2
from datetime import datetime

from components.logger import logger


class VideoProcessingTracker:
    def __init__(self, db_url=None):
        if db_url is None:
            db_url = os.environ.get('DATABASE_URL')

        if not db_url:
            raise ValueError("Database URL not provided and not found in environment variables")

        logger.info("Initializing VideoProcessingTracker with Heroku PostgreSQL")
        self.conn = psycopg2.connect(db_url)
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
            "SELECT status FROM video_processing_status WHERE video_id = %s",
            (video_id,)
        )
        result = cursor.fetchone()
        if result is None or result[0] != "completed":
            logger.info(f"Starting processing for video_id: {video_id}")
            cursor.execute(
                """
                INSERT INTO video_processing_status
                (video_id, status, start_time) VALUES (%s, %s, %s)
                ON CONFLICT (video_id) DO UPDATE
                SET status = EXCLUDED.status, start_time = EXCLUDED.start_time
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
            SET status = 'completed', end_time = %s
            WHERE video_id = %s
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
            SET status = 'failed', end_time = %s
            WHERE video_id = %s
            """,
            (datetime.now(), video_id),
        )
        self.conn.commit()

    def get_status(self, video_id):
        logger.debug(f"Getting status for video_id: {video_id}")
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT status FROM video_processing_status WHERE video_id = %s", (video_id,)
        )
        result = cursor.fetchone()
        logger.debug(f"Status for video_id {video_id}: {result[0] if result else None}")
        return result[0] if result else None

    def get_unprocessed_videos(self):
        logger.info("Getting unprocessed videos")
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT video_id FROM video_processing_status WHERE status != 'completed'"
        )
        result = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(result)} unprocessed videos")
        return result

    def check_if_video_exists_and_completed(self, video_id):
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT video_id, status FROM video_processing_status WHERE video_id = %s',
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
