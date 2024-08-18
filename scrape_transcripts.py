## We want to get all video URLs of a YouTube channel, these will be parsed to get video ID
## This video ID will then be used to scrape transcripts - repeat for all videos in the channel

import scrapetube
import os, getpass
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

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
def split_string_to_chunks(document):
    # Use list comprehension to split the string into chunks of 'chunk_size'
    return [document[i:i + 99999] for i in range(0, len(document))]

def embed_splits(doc, semantic_splitter):
    for large_chunk in split_string_to_chunks(doc.page_content):
                #TODO - Semantically split the documents, embed the splits, index
                if len(large_chunk) < 30:
                    continue
                doc = semantic_splitter.split_text(large_chunk)
                # Write the splits to document splits
                if len(doc) != 0:
                    vectorstore.add_texts(texts=doc)

# 88""Yb 888888 88""Yb .dP"Y8 88 .dP"Y8 888888 888888 88b 88  dP""b8 888888 
# 88__dP 88__   88__dP `Ybo." 88 `Ybo."   88   88__   88Yb88 dP   `" 88__   
# 88"""  88""   88"Yb  o.`Y8b 88 o.`Y8b   88   88""   88 Y88 Yb      88""   
# 88     888888 88  Yb 8bodP' 88 8bodP'   88   888888 88  Y8  YboodP 888888 



# .dP"Y8  dP""b8 88""Yb    db    88""Yb 88 88b 88  dP""b8       oo       888888 8b    d8 88""Yb 888888 8888b.  8888b.  88 88b 88  dP""b8 
# `Ybo." dP   `" 88__dP   dPYb   88__dP 88 88Yb88 dP   `"    ___88___    88__   88b  d88 88__dP 88__    8I  Yb  8I  Yb 88 88Yb88 dP   `" 
# o.`Y8b Yb      88"Yb   dP__Yb  88"""  88 88 Y88 Yb  "88    """88"""    88""   88YbdP88 88""Yb 88""    8I  dY  8I  dY 88 88 Y88 Yb  "88 
# 8bodP'  YboodP 88  Yb dP""""Yb 88     88 88  Y8  YboodP       ""       888888 88 YY 88 88oodP 888888 8888Y"  8888Y"  88 88  Y8  YboodP 

# Iterate over each video URL, load the transcript, and save to a separate file
languages = ["en-US", "en"] # We only want transcripts in these languages
for index, url in enumerate(video_urls):
    # Check if already processed - if it is don't need to repeat
    transcription_name = f"transcription{index+1}"
    if transcription_name in completed_transcriptions:
        continue

    # Check if has english transcript
    has_valid_transcript = False
    loader = YoutubeLoader.from_youtube_url(url)
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

    # Write the full transcript to a file, split it, then embed the splits
    embedding_model = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = Chroma("Prototype-Vector", embedding_model)

    # Semantic Splitter experimentation
    semantic_text_splitter = AI21SemanticTextSplitter()
    filename = f"transcription{index+1}.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        for doc in video_transcript:
            file.write(doc.page_content)

    # Take note that it's completed
    with open('completed_transcriptions.txt', 'a', encoding='utf-8') as log_file:
        log_file.write(f"transcription{index+1}: {url}\n")

    print(f"Saved transcript {index+1}")

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
