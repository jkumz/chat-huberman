import os

def __read_transcript_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def create_mock_scraper_output(transcript_file_path):
    transcript_content = __read_transcript_file(transcript_file_path)
    chunk_size = 99999
    return [{
        "url": "https://www.youtube.com/watch?v=pZX8ikmWvEU",
        "chunks": [transcript_content[i:i + chunk_size] for i in range(0, len(transcript_content), chunk_size)]
    }]

def get_sample_transcript_path():
    # Adjust this path to where your transcript file is located relative to the test file
    return os.path.join(os.path.dirname(__file__), '..', 'sample_transcript.txt')