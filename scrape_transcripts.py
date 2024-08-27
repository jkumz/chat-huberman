## We want to get all video URLs of a YouTube channel, these will be parsed to get video ID
## This video ID will then be used to scrape transcripts - repeat for all videos in the channel

from typing import List
import scrapetube
import os, getpass
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from yt_dlp import YoutubeDL
from bs4 import BeautifulSoup

from video_processing_tracker import VideoProcessingTracker

# Load API key
load_dotenv()
ai21_api_key = os.environ["AI21_API_KEY"]

videos = scrapetube.get_channel(channel_username="hubermanlab")
video_urls = []
document_splits = []

# iterate over scraped videos, get video URLs
for video in videos:
    url = "https://www.youtube.com/watch?v="+str(video['videoId'])
    video_urls.append(url)

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

# 88  88 888888 88     88""Yb 888888 88""Yb    8b    d8 888888 888888 88  88  dP"Yb  8888b.  .dP"Y8 
# 88  88 88__   88     88__dP 88__   88__dP    88b  d88 88__     88   88  88 dP   Yb  8I  Yb `Ybo." 
# 888888 88""   88  .o 88"""  88""   88"Yb     88YbdP88 88""     88   888888 Yb   dP  8I  dY o.`Y8b 
# 88  88 888888 88ood8 88     888888 88  Yb    88 YY 88 888888   88   88  88  YbodP  8888Y"  8bodP'

# Method to split a String to processable chunks - semantic text splitter can only work with 99,999 chars at a time
def prepare_document_chunks(document):
    # Use list comprehension to split the string into chunks of 'chunk_size'
    return [document[i:i + 99999] for i in range(0, len(document))]

def embed_splits(doc, semantic_splitter):
    for large_chunk in prepare_document_chunks(doc.page_content):
                #TODO - Semantically split the documents, embed the splits, index
                if len(large_chunk) < 30:
                    continue
                doc = semantic_splitter.split_text(large_chunk)
                # Write the splits to document splits
                if len(doc) != 0:
                    vectorstore.add_texts(texts=doc)

# Extracts URL and Title, returns dictionary
def extract_metadata(video_url):
    opts = {}
    with YoutubeDL(opts) as yt:
        info = yt.extract_info(url, download=False)
        data = {
            "url": video_url,
            "title": info.get("title")
        }
        return data

# Processes document chunks, embeds and indexes in pinecone
def process_and_index_chunks(doc_chunks: List[str], video_id: str, batch_size: int=100):
    tracker = VideoProcessingTracker()
    url = "https://www.youtube.com/watch?v="+str(video['videoId'])
    title = extract_metadata(url)["title"]

    try:
        tracker.start_processing(video_id)
        vectors_to_upsert = []

        for chunk_i, doc_chunk in enumerate(doc_chunks):
            if (len(doc_chunk) < 30):
                continue

            semantic_splits = semantic_text_splitter.split_text(doc_chunk)
            for split_i, split in enumerate(semantic_splits):
                embedding = embedding_model.embed_query(split)
                metadata = {
                    "video_id": video_id,
                    "video_url": url,
                    "video_title": title,
                    "chunk_index": chunk_i,
                    "split_index": split_i,
                    "text": split
                }

                vector_id = f"{video_id}_chunk{chunk_i}_split{split_i}"
                vectors_to_upsert.append((vector_id, embedding, metadata))

                if len(vectors_to_upsert) >= batch_size:
                    index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []

        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)

        tracker.complete_processing(video_id)
        print(f"Successfully processed and indexed video: {video_id}")

    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        tracker.fail_processing(video_id)

# 88""Yb 888888 88""Yb .dP"Y8 88 .dP"Y8 888888 888888 88b 88  dP""b8 888888 
# 88__dP 88__   88__dP `Ybo." 88 `Ybo."   88   88__   88Yb88 dP   `" 88__   
# 88"""  88""   88"Yb  o.`Y8b 88 o.`Y8b   88   88""   88 Y88 Yb      88""   
# 88     888888 88  Yb 8bodP' 88 8bodP'   88   888888 88  Y8  YboodP 888888 

from pinecone import Pinecone, ServerlessSpec
import time

# Initialise connection to Pinecone
pinecone_api_key = os.environ["PINECONE_API_KEY"]

# Configure Pinecone client
pinecone = Pinecone(api_key=pinecone_api_key, source_tag="pinecone-chathuberman")

# Define cloud region settings
region = os.environ["PINECONE_REGION"]
cloud = os.environ["PINECONE_CLOUD"]

# Define serverless specs
specs = ServerlessSpec(cloud=cloud, region=region)

# Define index
index_name = "chat-huberman"

# Check existing indexes - if it doesn't exist then create it
existing_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]
if index_name not in existing_indexes:
    # Create if doesn't exist
    pinecone.create_index(index_name, dimension=3072, spec=specs)
    while not pinecone.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print(f"Index {index_name} has been created")
else:
    print(f"Index {index_name} already exists - using existing index")

# .dP"Y8  dP""b8 88""Yb    db    88""Yb 88 88b 88  dP""b8       oo       888888 8b    d8 88""Yb 888888 8888b.  8888b.  88 88b 88  dP""b8 
# `Ybo." dP   `" 88__dP   dPYb   88__dP 88 88Yb88 dP   `"    ___88___    88__   88b  d88 88__dP 88__    8I  Yb  8I  Yb 88 88Yb88 dP   `" 
# o.`Y8b Yb      88"Yb   dP__Yb  88"""  88 88 Y88 Yb  "88    """88"""    88""   88YbdP88 88""Yb 88""    8I  dY  8I  dY 88 88 Y88 Yb  "88 
# 8bodP'  YboodP 88  Yb dP""""Yb 88     88 88  Y8  YboodP       ""       888888 88 YY 88 88oodP 888888 8888Y"  8888Y"  88 88  Y8  YboodP 

embedding_model = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"], model=os.environ["OPENAI_EMBEDDING_MODEL"])
vectorstore = Chroma("Prototype-Vector", embedding_model)
semantic_text_splitter = AI21SemanticTextSplitter()
index = pinecone.Index(host=os.environ["INDEX_HOST"], name=index_name)

# Iterate over each video URL, load the transcript, and save to a separate file
languages = ["en-US", "en"] # We only want transcripts in these languages
for index, url in enumerate(video_urls):
    # TODO - Check if already processed - if it is don't need to repeat - use metadata

    # Check if has english transcript
    has_valid_transcript = False
    loader = YoutubeLoader.from_youtube_url(url)
    video_id = ""
    for lang in languages:
        try:
            loader = YoutubeLoader.from_youtube_url(url, language=lang)
            video_transcript = loader.load()
            has_valid_transcript = True
            break
        except Exception as exception:
            print(f"Attempt with language '{lang}' failed: {exception}")
    if not has_valid_transcript:
        print(f"No suitable transcript found for video {index+1} - Skipping...")
        continue

    # Split the document semantically - semantic splitter can only chunk 99,999 chars at a time
    # so we use List Comprehension to prepare the chunks to be semantically split
    doc_chunks = [
        chunk
        for doc in video_transcript
        for chunk in prepare_document_chunks(doc.page_content)
    ]
    

    process_and_index_chunks(doc_chunks=doc_chunks, video_id=video_id)
    # When this is done, we must somehow take note of what entire video is completed
    # We must also implement some sort of handling for if it crashes mid index
    # either by re-indexing, or if it terminates and we've lost the embedding, then clearing the 
    # index for that video ID and doing again with fresh embeddings!

    print(f"Saved transcript {index+1}")

#     languages = ["en-US", "en"] # We only want transcripts in these languages
# for index, url in enumerate(video_urls):
#     # Check if already processed - if it is don't need to repeat
#     transcription_name = f"transcription{index+1}"
#     if transcription_name in completed_transcriptions:
#         continue

#     # Check if has english transcript
#     has_valid_transcript = False
#     loader = YoutubeLoader.from_youtube_url(url)
#     for lang in languages:
#         try:
#             loader = YoutubeLoader.from_youtube_url(url, language=lang)
#             video_transcript = loader.load()
#             has_valid_transcript = True
#             break
#         except Exception as exception:
#             print(f"Attempt with language '{lang}' failed: {exception}")
#     if not has_valid_transcript:
#         print(f"No suitable transcript found for video {index+1} - Skipping...")
#         continue

#     filename = f"transcription{index+1}.txt"
#     with open(filename, 'w', encoding='utf-8') as file:
#         for doc in video_transcript:
#             file.write(doc.page_content)

#     # Take note that it's completed
#     with open('completed_transcriptions.txt', 'a', encoding='utf-8') as log_file:
#         log_file.write(f"transcription{index+1}: {url}\n")

#     print(f"Saved transcript {index+1}")

## Testing
# from langchain.prompts import ChatPromptTemplate

# retriever = vectorstore.as_retriever()
# question = "What are the long term of effects of ADHD on dopamine levels?"

# context = retriever.invoke(question)

# template = """Answer the following question based only on the following context:
# {context}

# Question: {question}"""

# prompt = ChatPromptTemplate.from_template(template=template)

# llm = ChatOpenAI(model="gpt-4o", temperature=0)

# chain = prompt | llm

# chain.invoke({"context":context, "question": question})


## We also want to output each of the video transcriptions to a wider context window
## They come in splits already when we use the youtube loader
