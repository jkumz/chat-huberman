![ChatHuberman](https://github.com/user-attachments/assets/c3c505dc-5be8-405a-882e-eea1ea4ca961)

## A ChatGPT style chatbot based on the Huberman Lab podcast

This was a fun learning project I took upon myself while travelling. It implements Retrieval Augmented Generation (RAG) with some advanced techniques to provide scientific answers from world leading researchers and scientists in their respective fields.

ChatHuberman was split into three microservices and deployed on Heroku for a fast deployment and is accessible on https://www.chathuberman.xyz (may not be accessible in the future).

## Architecture

<img width="1316" alt="architecture" src="https://github.com/user-attachments/assets/974ea84a-d559-4ed8-81a1-b993b98b79f4">

## Demo
https://github.com/user-attachments/assets/1654fc31-d286-472d-a618-d20abdfd095b

## User Instructions
If you want to set this up using your own Pinecone index & microservices. The process for setting up a Pinecone index is relatively simple. You can do it using the UI and plug in your details into the environment as needed, or you could do it programatically.

The microservices were deployed on Heroku and if you wish to fully deploy them to get the same set up as in the demo, you will need to create three Heroku apps; a scraper-indexer app hosting the scraper-indexer microservice; a rag engine app hosting the rag engine microservice; a frontend app hosting the Streamlit frontend.

The scraper-indexer app comes with a PostgreSQL database for managing state of video processing status. This needs to be added on in your Heroku app and the URL to it passed in as an environment variable, or a separate PostgreSQL database of your choosing if you prefer that.

### Environment Variables

#### Scraper-Indexer 
- AI21_API_KEY
- LANGCHAIN_TRACING_V2=true
- LANGCHAIN_API_KEY
- LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
- LANGSMITH_API_KEY
- OPENAI_API_KEY
- USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
- PINECONE_API_KEY
- PINECONE_REGION
- PINECONE_CLOUD
- INDEX_HOST
- INDEX_NAME
- INDEX_SOURCE_TAG
- DATABASE_URL

#### RAG Engine
- LANGCHAIN_TRACING_V2=true
- LANGCHAIN_API_KEY
- LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
- OPENAI_API_KEY
- LOW_COST_LLM="gpt-4o-mini"
- CRITIC_LLM="gpt-4"
- ANTHROPIC_API_KEY
- PINECONE_API_KEY
- PINECONE_REGION
- PINECONE_CLOUD
- INDEX_HOST
- INDEX_NAME
- INDEX_SOURCE_TAG

#### Streamlit Frontend
- ENGINE_URL

### How to modify it for own purposes
This specific RAG pipeline uses the Huberman Lab podcast's youtube channel, but this code could very easily be modified to index and scrape any youtube channel and use it in a RAG pipeline. The main change would be changing the channel username and index details in the scraper-indexer (if you want separate indexes for separate channels).

It would also be relatively easy to extend and allow the user to decide which channel they want to do it to themselves and have it done through a dynamic process, rather than a channel statically defined in the code, although it takes some time to scrape, embed and index an entire youtube channel. This however goes out of the scope of this learning project.

## Metrics

Metrics were taken using the RAGAS framework, though the eval code would need some slight modifications now as I have changed the design of certain parts of the pipeline since metrics were taken to accommodate for a microservice architecture. Prior to that though the pipeline scored relatively high. The RAGAS scores can be found in rag_backend/eval/eval_results.csv

The final eval results are ranked between 0 and 1. (1 is the max)
- **Context Precision:** 0.885
	- Evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks.
- **Faithfulness:** 0.973
	- The ratio of number of claims in generated response from the given context to the total number of claims in the generated response.
- **Answer Relevance:** 0.955
	- Scores the relevancy of the answer according to the given question.
- **Context Recall:** 0.966
	- Measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth.
- **Answer Similarity:** 0.926
	- Scores the similarity between the answer and the ground truth.
- **Answer Correctness:** 0.673
	- Measures answer correctness compared to ground truth as a combination of factuality and semantic similarity.
	- An important note here is that the generated answer often covers the ground truth and goes further, which has an impact on this score which only compares to a ground truth.

More information on the RAGAS framework can be found [here](https://docs.ragas.io/en/stable/).

## Testing
Testing consisted of unit tests and integration tests where applicable. Each microservice was extensively tested individually, and a CI/CD pipeline is set up which automatically runs these tests with any updates to the GitHub repo.

## Drawbacks

- Performance could be improved using parallelisation
- Prompt engineering could be further enhance to limit token usage
- Anthropic's prompt caching could be further implemented to minimise system prompt token usage. At the time of this project it wasn't fully compatible with LangChain and the time vs reward to integrate it wasn't worth it for a learning project
- LangChain's YouTube Loader is good, but it's not perfect. To improve on this a transcript cleaning process could be added post transcription, before embedding and indexing
- Multi-query generation isn't always used, and when it's used is decided by an LLM based on a criteria set in a prompt. Due to the non-deterministic nature of the LLMs this tends to be biased towards using multi-query generation, but in the context of a neuroscience podcast it's certainly not bad, however if this was expanded to other sources this would have to be addressed.
- This uses LangChain's vector store as retriever functionality to provide good retrieval of documents, but a better but more long winded approach would be to build your own retriever. Specifically, a [Blender Retriever](https://arxiv.org/abs/2404.07220) would perform very well, but this goes beyond the scope of this learning project.

## References

[Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401)
- Foundation for RAG implementation

[Singh, S. (2024). "RAG Fusion: Enhancing Retrieved Context Quality Through Reciprocal Rank Fusion"](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1)
- Improved context retrieval & utilisation

[Yu, Z., et al. (2024). "Blender Retriever: A Novel Framework for Hybrid Dense-Sparse Document Retrieval"](https://arxiv.org/abs/2404.07220)
- A good reference for potential future retrieval improvements

## Implementation Resources

[Kamradt, G. (2024). "Semantic Splitting Strategies in RAG Systems"](https://www.youtube.com/watch?v=8OJC21T2SL4)
- Influenced text splitting strategy

[LangChain RAG From Scratch playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)
- Helped with understanding and implementation of each part of the RAG pipeline
