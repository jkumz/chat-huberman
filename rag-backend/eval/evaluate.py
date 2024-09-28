# We measure the following metrics
# - Faithfulness - The ratio of number of claims in generated response from the given context to the total number of claims in the generated response.
# - Answer Relevance - Scores the relevancy of the answer according to the given question.
# - Answer Similarity - Scores the similarity between the answer and the ground truth.
# - Answer Correctness - Measures answer correctness compared to ground truth as a combination of factuality and semantic similarity.
# - Context Precision - Evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. 
# - Context Recall - Measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth.

"""
IMPORTANT NOTE ON "ANSWER CORRECTNESS", "ANSWER RELEVANCY", "ANSWER SIMILARITY" METRICS

The "answer correctness" metric in our case (neuroscience podcast) is not as useful as it could be, due to the way it's calculated.
A ground truth answer is assigned to a question, and then an LLM checks how well aligned the response is to the ground truth.
Our responses usually cover a broad range of information, and while they may contain the answer, they also often include information beyond that.
The information beyond the simple answer can sometimes give the impression that the response is less aligned with the ground truth than it actually is.
The <thinking> tags also throw off the metric a little, as the content here is not relevant to the ground truth.
This also affects the answer relevance, where redundant details bring down the score, but these are the LLMs thoughts that the user doesn't normally see.
The answer similarity score may also be a little off - this compares embeddings between answer and ground truth. The LLM thoughts may again lower the score.
This is a consequence of Chain of Thought prompting. As of now RAGAS doesn't have a way to handle this, so we're stuck with it for now.

In general, the RAGAS score is a good metric to use, as it takes into account the context and the answer in a broader scope.
"""

# First we need to prepare the documents. At the time of writing this there is 28856 documents in the index.
# We aren't going to load all the documents as that's just inpractical, but we can load a sample from a specific
# transcript/video id. This way, we can provide ground truth comparisons.
# This should still result in over 100 documents, which are semantic splits.

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


#2 - PREPARE MODELS TO GENERATE SYNTHETIC QUESTION/ANSWER/GROUND TRUTH
generator_llm = ChatOpenAI(model=os.environ["LOW_COST_LLM"])
critic_llm = ChatOpenAI(model=os.environ["CRITIC_LLM"])
embeddings = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"], api_key=os.environ["OPENAI_API_KEY"])

# To save costs, should we re-use one generated test test? I suppose this offsets the effectiveness of the test set.
# So maybe not, but until it's set up and working we can do this to save costs, as every time we run this we 
# generate a new test set and cost money doing so.

async def main():
    #3 - GENERATE SYNTHETIC QUESTION/ANSWER/GROUND TRUTH
    generator = TestsetGenerator.from_langchain(generator_llm=generator_llm, critic_llm=critic_llm, embeddings=embeddings)
    testset = generator.generate_with_langchain_docs(documents=documents, test_size=10)

    #4 - STRUCTURE THE DATA FOR EVALUATION
    rag_engine = RAGEngine()
    eval_data = []
    for item in testset.test_data:
        rag_response = rag_engine.get_answer_with_context(item.question)
        rag_context = [doc.page_content for doc in rag_response["context"]]
        eval_data.append({
            "question": item.question,
            "answer": rag_response["answer"][0],
            "contexts": item.contexts, 
            "retrieved_contexts": rag_context,
            "ground_truths": [item.ground_truth],
            "reference": item.ground_truth,
        })
    dataframe = pd.DataFrame(eval_data)
    final_dataset = Dataset.from_pandas(dataframe)

    #5 - EVALUATE THE FINAL DATASET / TESTSET USING RAGAS
    result = evaluate(
        final_dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_similarity,
            answer_correctness,
        ],
    )

    #6 - STORE THE EVALUATION RESULTS
    # Create a DataFrame with all the evaluation data and results
    timestamp = pd.Timestamp.now()
    result_df = pd.DataFrame({
        'context_precision': [result['context_precision']],
        'faithfulness': [result['faithfulness']],
        'answer_relevancy': [result['answer_relevancy']],
        'context_recall': [result['context_recall']],
        'answer_similarity': [result['answer_similarity']],
        'answer_correctness': [result['answer_correctness']],
        'timestamp': [timestamp],
        'questions': [[item['question'] for item in eval_data]]
    })


    # Define the output CSV file
    results_csv_file = "rag-backend/eval/eval_results.csv"

    # Append the results to the CSV file
    result_df.to_csv(results_csv_file, mode='a', header=not os.path.exists(results_csv_file), index=False)

    # Store dataset used in evaluation in a txt file, timestamps correspond to the timestamp of the evaluation
    dataset_txt_file = "rag-backend/eval/eval_dataset.txt"
    with open(dataset_txt_file, 'a') as f:
        for item in eval_data:
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Answer: {item['answer']}\n\n")
            f.write(f"Contexts: {str(item['contexts'])}\n\n")
            f.write(f"Retrieved Contexts: {str(item['retrieved_contexts'])}\n\n")
            f.write(f"Ground Truths: {str(item['ground_truths'])}\n\n")


    print(f"Evaluation results appended to {results_csv_file}")
    print(f"Dataset appended to {dataset_txt_file}")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
