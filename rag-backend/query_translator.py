from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from prompts import get_multi_query_generation_prompt, get_check_if_multi_query_should_be_used_prompt
import os


class QueryTranslator:
    # We only need a low cost LLM as it is only used for generating alternative queries
    def __init__(self):
        self.llm = ChatOpenAI(model=os.getenv("LOW_COST_LLM"), api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0)

    '''
        This method is used to determine if the user query would benefit from multi-query generation.
    '''
    def should_use_multi_query(self, query):
        prompt_template = get_check_if_multi_query_should_be_used_prompt()
        
        decision_chain = prompt_template | self.llm | StrOutputParser()
        decision = decision_chain.invoke({"query": query}).strip().lower()
        
        return decision == "yes"

    '''
        This method is used to generate multiple queries from a user input.
        The alternative queries are used to retrieve more relevant documents from a vector database.
    '''
    def multi_query_generation(self):
        prompt_template = get_multi_query_generation_prompt()
        # We want to add the RAG Fusion reranking step here, so documents get reranked before being sent to the user
        generate_queries_chain = prompt_template | self.llm | StrOutputParser() | (lambda x: x.split("\n"))
        return generate_queries_chain

    '''
        Helper method in multi query generation. Goes through the list of lists of retrieved documents,
        merges the lists and removes duplicates.
    '''
    def get_unique_union(self, retrieved_docs):
        # Flatten list of lists
        list_all_docs = [doc for sublist in retrieved_docs for doc in sublist]
        # Create a set of unique tuples representing each document
        unique_doc_tuples = set(self.serialise_doc(doc=doc) for doc in list_all_docs)
        # Convert tuples back to Document objects
        unique_docs = [
            self.deserialise_doc(doc=t) for t in unique_doc_tuples
        ]
        return unique_docs

    '''
        Reciprocal Rank Fusion is a method of reranking a list of documents based on their relevance to a user query.
        It is used to overcome the limitations of distance-based similarity search, where documents that are similar
        in content may be far apart in the vector space.
    '''
    def reciprocal_rank_fusion(self, result_docs, k = 60):

        fused_scores = {}

        # Iterate through each list of ranked results List[Document(Metadata dictionary, PageContent)]
        # Apply Reciprocal Rank Fusion formula to each list of documents, re-ranking the documents
        for docs in result_docs:
            for rank, doc in enumerate(docs):
                # Convert doc to tuple, use as key
                key = self.serialise_doc(doc=doc)
                # If no score yet, set to 0
                if key not in fused_scores:
                    fused_scores[key] = 0
                # Reciprocal Rank Fusion formula
                prev_score = fused_scores[key]
                fused_scores[key] = prev_score + 1 / (rank + k)

        # Sort documents based on fused scores, highest score ran
        # The fused score dictionary removes duplicates by using repetitions in the key to improve score
        # so no need to perform unique union operation
        reranked_results = [
            ((doc), score) for doc, score in sorted(fused_scores.items(), key = lambda x: x[1], reverse = True)
        ]

        # Convert back to Document objects
        return [
            self.deserialise_doc(doc=doc) for (doc, score) in reranked_results
        ]

# Decided not to serialise/deserialise to JSON as had no real use for it, only used in query translations

    '''
        Serialises a document to a tuple.
    '''
    def serialise_doc(self, doc):
        return (
            doc.metadata.get('chunk_index'),
            doc.metadata.get('split_index'),
            doc.metadata.get('video_id'),
            doc.metadata.get('video_title'),
            doc.metadata.get('video_url'),
            doc.page_content
        )

    '''
        Deserialises a document from a tuple to a Document object.
    '''
    def deserialise_doc(self, doc):
        return Document(
            metadata={
                'chunk_index': doc[0],
                'split_index': doc[1],
                'video_id': doc[2],
                'video_title': doc[3],
                'video_url': doc[4]
            },
            page_content=doc[5]
        )