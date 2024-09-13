from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.load import dumps, loads

import os


class QueryTranslator:
    def __init__(self):
        self.llm = ChatOpenAI(model=os.getenv("LOW_COST_LLM"), api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0)

    '''
        This method is used to determine if the user query would benefit from multi-query generation.
    '''
    def should_use_multi_query(self, query):
        prompt_template = ChatPromptTemplate.from_template(
            """
            Analyze the following user query and determine if it would benefit from multi-query generation.
            Consider the following factors:
            1. Vagueness: Is the query too broad or unclear?
            2. Complexity: Does the query involve multiple concepts or require a nuanced understanding?
            3. Ambiguity: Could the query be interpreted in multiple ways?
            4. Lack of context: Is there missing information that could lead to multiple interpretations?

            If the query would benefit from multi-query generation, respond with "Yes". Otherwise, respond with "No".

            User query: {query}

            Decision (Yes/No):
            """
        )
        
        decision_chain = prompt_template | self.llm | StrOutputParser()
        decision = decision_chain.invoke({"query": query}).strip().lower()
        
        return decision == "yes"

    '''
        This method is used to generate multiple queries from a user input.
        The alternative queries are used to retrieve more relevant documents from a vector database.
    '''
    def multi_query_generation(self):
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an expert at generating multiple queries from a user input.
            Your job is to generate 5 different versions of the given user question to retrieve
            relevant documents from a vector database. By generating multiple queries, your goal is to
            help the user overcome some of the limitations of distanced-based similarty search.
            Provide tehse alternative questions separated by new lines. 
            Original question: {user_input}            
            """
        )
        generate_queries_chain = prompt_template | self.llm | StrOutputParser() | (lambda x: x.split("\n"))
        return generate_queries_chain
    '''
        Helper method in multi query generation. Goes through the list of lists of retrieved documents,
        merges the lists, and removes duplicates.
    '''
    def get_unique_union(self, retrieved_docs):
        # Flatten list of lists
        list_all_docs = [doc for sublist in retrieved_docs for doc in sublist]
        # Create a set of unique tuples representing each document
        unique_doc_tuples = set((
            doc.metadata.get('chunk_index'),
            doc.metadata.get('split_index'),
            doc.metadata.get('video_id'),
            doc.metadata.get('video_title'),
            doc.metadata.get('video_url'),
            doc.page_content
        ) for doc in list_all_docs)
        # Convert tuples back to Document objects
        unique_docs = [
            Document(
                metadata={
                    'chunk_index': t[0],
                    'split_index': t[1],
                    'video_id': t[2],
                    'video_title': t[3],
                    'video_url': t[4]
                },
                page_content=t[5]
            ) for t in unique_doc_tuples
        ]
        return unique_docs
    
