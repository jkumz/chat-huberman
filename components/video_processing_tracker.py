import os
import sqlite3
from datetime import datetime
from typing import List

class VideoProcessingTracker:
# Find the video_processing.db file in the /persistence folder
    def find_db_path(self, max_depth=10):
        current_dir = os.getcwd()
        for _ in range(max_depth):
            if os.path.exists(os.path.join(current_dir, 'persistence', 'video_processing.db')):
                return os.path.join(current_dir, 'persistence', 'video_processing.db')
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached the root directory
                break
            current_dir = parent_dir
        raise FileNotFoundError("Could not find the video_processing.db file in the /persistence directory at the project root")

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = self.find_db_path()
        
        self.conn = sqlite3.connect(db_path)
        self.create_table()

# Create table for tracking video processing status
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_processing_status
        (video_id TEXT PRIMARY KEY, 
         status TEXT, 
         start_time TIMESTAMP,
         end_time TIMESTAMP)
        ''')
        self.conn.commit()

# Start processing video - insert relevant rows
    def start_processing(self, video_id):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO video_processing_status
        (video_id, status, start_time) VALUES (?, ?, ?)
        ''', (video_id, 'processing', datetime.now()))
        self.conn.commit()

# Update relevant rows upon completion
    def complete_processing(self, video_id):
        cursor = self.conn.cursor()
        cursor.execute('''
        UPDATE video_processing_status
        SET status = 'completed', end_time = ?
        WHERE video_id = ?
        ''', (datetime.now(), video_id))
        self.conn.commit()

# Mark as failed so that we can restart if failed
    def fail_processing(self, video_id):
        cursor = self.conn.cursor()
        cursor.execute('''
        UPDATE video_processing_status
        SET status = 'failed', end_time = ?
        WHERE video_id = ?
        ''', (datetime.now(), video_id))
        self.conn.commit()

    def get_status(self, video_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT status FROM video_processing_status WHERE video_id = ?', (video_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_unprocessed_videos(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT video_id FROM video_processing_status WHERE status != "completed"')
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        self.conn.close()