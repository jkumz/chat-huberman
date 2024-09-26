# First we need to prepare the documents. At the time of writing this there is 28856 documents in the index.
# We aren't going to load all the documents as that's just inpractical, but we can load a sample from a specific
# transcript/video id. This way, we can provide ground truth comparisons.
# This should still result in over 100 documents, which are semantic splits.

# We want to measure some of the following metrics, where relevant
# - RAGAS Score
# - Faithfulness - The ratio of number of claims in generated response from the given context to the total number of claims in the generated response.
# - Answer Relevance - Scores the relevancy of the answer according to the given question.
# - Answer Similarity - Scores the similarity between the answer and the ground truth.
# - Answer Correctness - Measures answer correctness compared to ground truth as a combination of factuality and semantic similarity.
# - Context Precision - Evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. 
# - Context Utilisation - Evaluates whether all of the answer relevant items present in the contexts are ranked higher or not.
# - Context Recall - Measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth.
# - Noise Sensitivity - Measures how often a system makes errors by providing incorrect responses when utilizing either relevant or irrelevant retrieved documents.
# - Aspect Critique - A binary classification assessing whether the submissions are correct (in our case)

# 1 - GET DOCS TO GENERATE SYNTHETIC Q/A/GT
import asyncio
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone
from dotenv import load_dotenv, find_dotenv
from datasets import Dataset
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness,
)
from ragas import evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag_engine import RAGEngine

load_dotenv(find_dotenv(filename=".rag_engine.env"))


pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(host=os.environ["INDEX_HOST"], name=os.environ["INDEX_NAME"])
embeddings = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"], api_key=os.environ["OPENAI_API_KEY"])
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
video_id = "csubiPlvFWk" # Dr. Jonathan Haidt: How Smartphones & Social Media Impact Mental Health & the Realistic Solutions

# Iterate over the index, and get the documents where the id starts with the video_id
ids = []
for id_list in index.list(prefix=video_id, limit=100):
    ids.extend(id_list)
query_results = [index.query(id=id, include_values=False, include_metadata=True, top_k=1) for id in ids]
documents = []
# Build the documents
for result in query_results:
    if result['matches']:
        match = result['matches'][0]
        metadata = match['metadata'].copy()
        video_id = metadata.pop('video_id', '')
        text = metadata.pop('text', '')
        document = Document(
            page_content=text,
            metadata={
                **metadata,
                'filename': video_id
            }
        )
        documents.append(document)


# GENERATE SYNTHETIC QUESTION/ANSWER/GROUND TRUTH
generator_llm = ChatOpenAI(model=os.environ["LOW_COST_LLM"])
critic_llm = ChatOpenAI(model=os.environ["CRITIC_LLM"])
embeddings = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"], api_key=os.environ["OPENAI_API_KEY"])

# To save costs, should we re-use one generated test test? I suppose this offsets the effectiveness of the test set.
# So maybe not, but until it's set up and working we can do this to save costs, as every time we run this we 
# generate a new test set and cost money doing so.

async def main():
    generator = TestsetGenerator.from_langchain(generator_llm=generator_llm, critic_llm=critic_llm, embeddings=embeddings)

    testset = generator.generate_with_langchain_docs(documents=documents, test_size=10)
    rag_engine = RAGEngine()
    eval_data = []
    for item in testset.test_data:
        rag_response = rag_engine.get_answer_with_context(item.question)
        rag_context = [doc.page_content for doc in rag_response["context"]]
        eval_data.append({
            "question": item.question,
            "answer": rag_response["answer"],
            "contexts": item.contexts, 
            "retrieved_contexts": rag_context,
            "ground_truths": [item.ground_truth],
            "reference": item.ground_truth,
        })
    dataframe = pd.DataFrame(eval_data)
    
    final_dataset = Dataset.from_pandas(dataframe)

    # EVALUATE
    # Okay, so for each of our metrics we need to go through their docs and find
    # which columns are missing for each one, as they aren't fully compatible.
    result = evaluate(
        final_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            answer_similarity,
            answer_correctness,
        ],
    )

    # Create a DataFrame with all the evaluation data and results
    #TODO Change CSV file so scores don't repeat 
    result_df = pd.DataFrame([{
        'context_precision': result['context_precision'],
        'faithfulness': result['faithfulness'],
        'answer_relevancy': result['answer_relevancy'],
        'context_recall': result['context_recall'],
        'answer_similarity': result['answer_similarity'],
        'answer_correctness': result['answer_correctness'],
        'question': item['question'],
        'answer': item['answer'],
        'contexts': str(item['contexts']),  # Convert list to string
        'retrieved_contexts': str(item['retrieved_contexts']),  # Convert list to string
        'ground_truths': str(item['ground_truths']),  # Convert list to string
        'reference': item['reference'],
        'timestamp': pd.Timestamp.now()
    } for item in eval_data])


    # Define the output CSV file
    csv_file = "rag-backend/eval/eval_results.csv"

    # Append the results to the CSV file
    result_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    print(f"Evaluation results appended to {csv_file}")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
