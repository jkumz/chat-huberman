import sqlite3
from datetime import datetime

class VideoProcessingTracker:
    def __init__(self, db_path='video_processing.db'):
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

# Usage in the main processing function
def process_and_index_chunks(doc_chunks: List[str], video_id: str, batch_size: int = 100):
    tracker = VideoProcessingTracker()
    tracker.start_processing(video_id)
    
    try:
        # Your existing processing code here
        # ...

        tracker.complete_processing(video_id)
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        tracker.fail_processing(video_id)
    finally:
        tracker.close()

# Example of how to use the tracker
tracker = VideoProcessingTracker()
unprocessed_videos = tracker.get_unprocessed_videos()
for video_id in unprocessed_videos:
    # Process the video
    process_and_index_chunks(get_doc_chunks_for_video(video_id), video_id)
tracker.close()