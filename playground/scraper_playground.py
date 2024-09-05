import scrapetube
import os
from yt_dlp import YoutubeDL
from langchain_openai import OpenAIEmbeddings
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from pinecone import Pinecone, ServerlessSpec
import time
from components.video_processing_tracker import VideoProcessingTracker



videos = scrapetube.get_channel(channel_username="hubermanlab")
video_urls = []
document_splits = []


# Extracts URL and Title, returns dictionary
def extract_metadata(video_url):
    opts = {}
    with YoutubeDL(opts) as yt:
        info = yt.extract_info(url, download=False)
        data = {"url": video_url, "title": info.get("title")}
        return data


# iterate over scraped videos, get video URLs
for video in videos:
    url = "https://www.youtube.com/watch?v=" + str(video["videoId"])
    video_urls.append(url)
    # light metadata scrape test
    # metadata = extract_metadata(url)
    # print(metadata)


print(f"Count of video URLs: {len(video_urls)}")

# Method to split a String to processable chunks - semantic text splitter can only work with 99,999 chars at a time
def prepare_document_chunks(document):
    # Use list comprehension to split the string into chunks of 'chunk_size'
    return [document[i : i + 99999] for i in range(0, len(document))]


### basically, the only metadata from the youtube loader is the video ID, so if we want
### more metadata we have to manually extract it for each URL
### we can give each semantic chunk it's video ID as well as youtube video title in the
### metadata, but should we bother? good extensibiltiy in future

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

embedding_model = OpenAIEmbeddings(
    api_key=os.environ["OPENAI_API_KEY"], model=os.environ["OPENAI_EMBEDDING_MODEL"]
)
vectorstore = Chroma("Prototype-Vector", embedding_model)
semantic_text_splitter = AI21SemanticTextSplitter()


tracker = VideoProcessingTracker()

# Iterate over each video URL, load the transcript, and save to a separate file
languages = ["en-US", "en"]  # We only want transcripts in these languages
for index, url in enumerate(video_urls):
    # TODO - Check if already processed - if it is don't need to repeat - use metadata
    tracker.start_processing(video_id)

    # Check if has english transcript
    has_valid_transcript = False
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

    # Get metadata of video - this will be in metadata of each semantic split for video
    # video_id = doc.metadata.get("source")

    # Split the document semantically - semantic splitter can only chunk 99,999 chars at a time
    # so we use List Comprehension to prepare the chunks to be semantically split
    doc_chunks = [
        chunk
        for doc in video_transcript
        for chunk in prepare_document_chunks(doc.page_content)
    ]

    ## Now semantically split all large chunks, go over each split, embed and store in index
    for chunk_i, doc_chunk in enumerate(doc_chunks):
        if len(doc_chunk) < 30:
            continue
        semantic_splits = semantic_text_splitter.split_text(doc_chunk)

        for split_i, split in enumerate(semantic_splits):
            embedding = embedding_model.embed_query(split)
            metadata = {
                "video_id": video_id,
                "title": "",  # TODO - Get title of video using Video ID
                "chunk_index": chunk_i,
                "split_index": split_i,
                "text": split,
            }

        vector_id = f"{video_id}_chunk{chunk_i}_split{split_i}"

        # Index the embeddings, add meta data - doc_chunk as content? - and then video ID
        pinecone
    # When this is done, we must somehow take note of what entire video is completed
    # We must also implement some sort of handling for if it crashes mid index
    # either by re-indexing, or if it terminates and we've lost the embedding, then clearing the
    # index for that video ID and doing again with fresh embeddings!

    print(f"Saved transcript {index+1}")
