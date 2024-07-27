## We want to get all video URLs of a YouTube channel, these will be parsed to get video ID
## This video ID will then be used to scrape transcripts - repeat for all videos in the channel

import scrapetube

videos = scrapetube.get_channel(channel_username="hubermanlab")
video_urls = []

for video in videos:
    id = "https://www.youtube.com/watch?v="+str(video['videoId'])
    video_urls.append(id)

print(f"Count of video URLs: {len(video_urls)}")

## At this stage, we've got the Video URLs of the entire youtube channel
## We want to use LangChain's Youtube Loader to load a list of lists of documents

from langchain_community.document_loaders import YoutubeLoader

# TODO: Add error code handling, retries etc
# TODO: Deploy this as a microservice on the web, polling and updating results daily in cloud
# Load existing logs to avoid reprocessing videos
completed_transcriptions = set()
try:
    with open('completed_transcriptions.txt', 'r', encoding='utf-8') as log_file:
        completed_transcriptions = {line.strip().split(': ')[0] for line in log_file}
except FileNotFoundError:
    print("Log file not found. A new one will be created.")

languages = ["en-US", "en"] 
# Iterate over each video URL, load the transcript, and save to a separate file
for index, url in enumerate(video_urls):
    # Check if already processed - if it is don't need to repeat
    transcription_name = f"transcription{index+1}"
    if transcription_name in completed_transcriptions:
        continue

    has_valid_transcript = False
    loader = YoutubeLoader.from_youtube_url(url)
    for lang in languages:
        try:
            loader = YoutubeLoader.from_youtube_url(url, language=lang)
            video_transcript = loader.load()
            has_valid_transcript = True
            break
        except Exception as invalid:
            print(f"Attempt with language '{lang}' failed: {invalid}")

    if not has_valid_transcript:
        print(f"No suitable transcript found for video {index+1} - Skipping...")
        continue

    # Write the full transcript to a file
    filename = f"transcription{index+1}.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        for doc in video_transcript:
            file.write(doc.page_content)

    # Take note that it's completed
    with open('completed_transcriptions.txt', 'a', encoding='utf-8') as log_file:
        log_file.write(f"transcription{index+1}: {url}\n")

    print(f"Saved transcript {index+1}")

## We also want to output each of the video transcriptions to a wider context window
## They come in splits already when we use the youtube loader

#print(f"Number of video transcripts: {len(transcripts)}")
#print(f"Transcript 1: {transcripts[0]}")