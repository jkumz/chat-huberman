import re
import sys
import os
import tiktoken
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import (load_dotenv, find_dotenv)
from rag_backend.query_translator import QueryTranslator
from rag_backend.prompts import get_main_prompt, get_few_shot_prompt
from concurrent.futures import TimeoutError
from rag_backend.logger import logger as logger


load_dotenv(find_dotenv(filename=".rag_engine.env"))

EMBEDDING_MODEL="text-embedding-3-large"

CLAUDE_SONNET_MODEL="claude-3-5-sonnet-20240620" # most powerful
SONNET_INPUT_COST_PER_TOKEN = 0.000003 # $3 per 1m tokens in
SONNET_OUTPUT_COST_PER_TOKEN = 0.000015 # $15 per 1m tokens out

CLAUDE_HAIKU_MODEL="claude-3-haiku-20240307" # less powerful
HAIKU_INPUT_COST_PER_TOKEN = 0.00000025 # $0.25 per 1m tokens in
HAIKU_OUTPUT_COST_PER_TOKEN = 0.00000125 # $1.25 per 1m tokens out

CLAUDE_OPUS_MODEL="claude-3-opus-20240229" # most performant
OPUS_INPUT_COST_PER_TOKEN = 0.000015 # $15 per 1m tokens in
OPUS_OUTPUT_COST_PER_TOKEN = 0.000075 # $75 per 1m tokens out

#TODO: Calculate cost per query after the fact, display to user
#TODO: Memory: Remember previous questions and answers in chat, and use them to inform the current answer
#TODO: Ability to go back to previous chats - MAYBE - this would require storing it in a database with user id etc, good learning experience though

class RAGEngine:
    """
    Constructor class for the RAG engine class
    
    Parameters:
    - index_name: The name of the Pinecone index to use
    - llm: The LLM to use for generating responses
    - embedding_model: The embedding model to use for generating embeddings
    - output_parser: The output parser to use for parsing the output of the LLM
    """
    def __init__(self, openai_api_key, anthropic_api_key, model=CLAUDE_SONNET_MODEL) :
        self.embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_api_key)
        self.anthropic_api_key = anthropic_api_key
        self.index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(name=os.getenv("INDEX_NAME"), host=os.getenv("INDEX_HOST"))
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self._set_model(model)
        self.output_parser = StrOutputParser()
        self.query_translator = QueryTranslator(openai_api_key=openai_api_key)

        self.generation_cost = 0.00
        self.retrieval_cost = 0.00
        self.translation_cost = 0.00
    '''
    Method for setting the model and updating the costs
    
    Parameters:
    - model: The model to use for generating responses
    ''' 
    def _set_model(self, model):
        self.model = model
        self.llm = ChatAnthropic(model=model, temperature=0, api_key=self.anthropic_api_key, model_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}})
        self.tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        self.__update_costs()

    '''
    Method for updating the costs
    '''
    def __update_costs(self):
        if self.model == CLAUDE_SONNET_MODEL:
            self.input_cost_per_token = SONNET_INPUT_COST_PER_TOKEN
            self.output_cost_per_token = SONNET_OUTPUT_COST_PER_TOKEN
        elif self.model == CLAUDE_HAIKU_MODEL:
            self.input_cost_per_token = HAIKU_INPUT_COST_PER_TOKEN
            self.output_cost_per_token = HAIKU_OUTPUT_COST_PER_TOKEN
        elif self.model == CLAUDE_OPUS_MODEL:
            self.input_cost_per_token = OPUS_INPUT_COST_PER_TOKEN
            self.output_cost_per_token = OPUS_OUTPUT_COST_PER_TOKEN
        else:
            raise ValueError(f"Unknown model: {self.model}")
        
    '''
    Method for calculating the cost of the input and output
    
    Parameters:
    - input_text: The input text to calculate the cost of
    - output_text: The output text to calculate the cost of
    
    Returns:
    - The cost of the input and output
    '''
    def _calculate_generation_cost(self, input_tokens, output_tokens):
        input_cost = input_tokens * self.input_cost_per_token
        output_cost = output_tokens * self.output_cost_per_token
        total_cost = input_cost + output_cost
        return total_cost

    '''
    Method for resetting the generation cost
    '''
    def _reset_generation_cost(self):
        self.generation_cost = 0.00

    '''
    Method for resetting the retrieval cost
    '''
    def _reset_retrieval_cost(self):
        self.retrieval_cost = 0.00

    '''
    Method for resetting the translation cost
    '''
    def _reset_translation_cost(self):
        self.translation_cost = 0.00

    '''
    Method for resetting all the costs
    '''
    def reset_all_costs(self):
        self._reset_generation_cost()
        self._reset_retrieval_cost()
        self._reset_translation_cost()

    def get_total_cost(self):
        return self.generation_cost + self.retrieval_cost + self.translation_cost
    '''
    Method for chaining together the components of the RAG engine
    
    Parameters:
    - user_input: The user's input query/question which needs answering
    - context: The context to use for generating the response
    
    Returns:
    - The response from the LLM
    '''
    def _chain(self, user_input, context, chat_history="", few_shot=False):
        if few_shot:
            prompt = get_few_shot_prompt()
        else:
            prompt = get_main_prompt()

        chain = prompt | self.llm
        response = chain.invoke({"question": user_input, "documents": context, "chat_history": chat_history})
        resp_metadata = response.usage_metadata
        parsed_response = self.output_parser.invoke(response.content)
        input_tokens = resp_metadata["input_tokens"]
        output_tokens = resp_metadata["output_tokens"]
        gen_cost = self._calculate_generation_cost(input_tokens, output_tokens)
        self.generation_cost += gen_cost
        self.translation_cost = self.query_translator.get_total_cost()
        return parsed_response
    
    '''
    Method for retrieving the most relevant chunks from the index
    
    Parameters:
    - user_input: The user's input query/question which needs answering
    
    Returns:
    - The most relevant chunks from the index
    ''' 
    #TODO - Calculate embedding cost of multi query prompts
    def retrieve_relevant_documents(self, user_input, use_reranking=True, timeout=30):
        async def retrieval_with_timeout():
            if self.query_translator.should_use_multi_query(user_input):
                # RAG Fusion method removes duplicates when reranking so no need for unique union in chain
                if use_reranking:
                    reranked_retrieval_chain = (
                        self.query_translator.multi_query_generation()
                        | self.retriever.map()
                    )
                    unflattened_unranked_docs = await asyncio.wait_for(
                        asyncio.to_thread(reranked_retrieval_chain.invoke, {"user_input": user_input}),
                        timeout=timeout
                    )
                    return await asyncio.wait_for(
                        asyncio.to_thread(self.query_translator.reciprocal_rank_fusion, result_docs=unflattened_unranked_docs),
                        timeout=timeout
                    )
                else:
                    retrieval_chain = (
                        self.query_translator.multi_query_generation()
                        | self.retriever.map()
                        | (lambda docs: self.query_translator.get_unique_union(docs))
                    )
                    return await asyncio.wait_for(
                        asyncio.to_thread(retrieval_chain.invoke, {"user_input": user_input}),
                        timeout=timeout
                    )
            else:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.retriever.invoke, user_input),
                    timeout=timeout
                )

        try:
            return asyncio.run(retrieval_with_timeout())
        except TimeoutError:
            raise TimeoutError(f"Retrieval operation timed out after {timeout} seconds") from None

    '''
    Method for getting the answer to a user's question

    Parameters:
    - user_input: The user's input query/question which needs answering

    Returns:
    - The answer to the user's question
    '''
    def get_answer(self, user_input, few_shot=False, format_response=True, history=""):
        retrieved = self.retrieve_relevant_documents(user_input)
        if format_response:
            raw = self._chain(user_input=user_input, context=retrieved, few_shot=few_shot, chat_history=history)
            return re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL)
        else:
            return self._chain(user_input=user_input, context=retrieved, few_shot=few_shot, chat_history=history)

    '''
    Method for getting the answer to a user's question along with the relevant context

    Parameters:
    - user_input: The user's input query/question which needs answering

    Returns:
    - The answer to the user's question along with the relevant context
    '''
    def get_answer_with_context(self, user_input, few_shot=False):
        retrieved = self.retrieve_relevant_documents(user_input)
        return {"answer": self._chain(user_input=user_input, context=retrieved, few_shot=few_shot), "context": retrieved}