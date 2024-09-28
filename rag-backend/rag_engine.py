from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from collections import deque
from pinecone import Pinecone
from dotenv import load_dotenv, find_dotenv
from query_translator import QueryTranslator
from prompts import get_main_prompt
import os
import tiktoken
from logger import logger


load_dotenv(find_dotenv(filename=".rag_engine.env"))

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
    def __init__(self, model=CLAUDE_SONNET_MODEL):
        self.embedding_model = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
        self.index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(name=os.getenv("INDEX_NAME"), host=os.getenv("INDEX_HOST"))
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self.set_model(model)
        self.output_parser = StrOutputParser()
        self.query_translator = QueryTranslator()
    '''
    Method for setting the model and updating the costs
    
    Parameters:
    - model: The model to use for generating responses
    ''' 
    def set_model(self, model):
        self.model = model
        self.llm = ChatAnthropic(model=model, temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.tokenizer = tiktoken.encoding_for_model(os.getenv("EMBEDDING_MODEL"))
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
    #TODO: Fix this, there is a discrepancy between the number of tokens in Claude backend and what is returned from the tokenizer
    def calculate_generation_cost(self, input_tokens, output_tokens):
        input_cost = input_tokens * self.input_cost_per_token
        output_cost = output_tokens * self.output_cost_per_token
        total_cost = input_cost + output_cost
        return input_cost, output_cost, total_cost

    '''
    Method for chaining together the components of the RAG engine
    
    Parameters:
    - user_input: The user's input query/question which needs answering
    - context: The context to use for generating the response
    
    Returns:
    - The response from the LLM
    '''
        
    # TODO: We can also look through the context, and if there is context that is not fully completed, we can
    # generate a query that will retrieve more relevant context and add it to the context list
    # Example: "Implementing the big 6 pillars of health: <6 pillars not mentioned, so LLM defines
    # a query that will retrieve more context on the 6 pillars and add it to the context list>"

        # TODO: Few shot learning: Give examples of previous conversations to the LLM to help it answer the question
        # - No "Based on the context provided" or "The context provided is" in the answer
    # - No "I used the context provided to answer your question" in the answer
    # - Sound like a human, not an AI
    def chain(self, user_input, context, chat_history=""):
        prompt = get_main_prompt()

        chain = prompt | self.llm
        response = chain.invoke({"question": user_input, "documents": context, "chat_history": chat_history})
        resp_metadata = response.usage_metadata
        parsed_response = self.output_parser.invoke(response.content)
        return parsed_response, resp_metadata
    
    '''
    Method for retrieving the most relevant chunks from the index
    
    Parameters:
    - user_input: The user's input query/question which needs answering
    
    Returns:
    - The most relevant chunks from the index
    ''' 
    # Most of the following ideas would be using a cheaper LLM to save costs as Claude 3.5 Sonnet is very expensive
    # Decided not to implement HyDE as running RAGAS has shown that retrieval is highly effective, no need for HyDE in this case.
    # Or other forms of query translation/transformation, as RAGAS results show there is no need
    
    def retrieve_relevant_documents(self, user_input, use_reranking=True):
        if self.query_translator.should_use_multi_query(user_input):
            # RAG Fusion method removes duplicates when reranking so no need for unique union in chain
            if use_reranking:
                reranked_retrieval_chain = (
                    self.query_translator.multi_query_generation()
                    | self.retriever.map()
                )
                unflattened_unranked_docs = reranked_retrieval_chain.invoke({"user_input": user_input})
                return self.query_translator.reciprocal_rank_fusion(result_docs=unflattened_unranked_docs)
            else:
                retrieval_chain = (
                    self.query_translator.multi_query_generation()
                    | self.retriever.map()
                    | (lambda docs: self.query_translator.get_unique_union(docs))
                )
                return retrieval_chain.invoke({"user_input": user_input})
        else:
            return self.retriever.invoke({"user_input": user_input})

    '''
    Method for getting the answer to a user's question

    Parameters:
    - user_input: The user's input query/question which needs answering

    Returns:
    - The answer to the user's question
    '''
    def get_answer(self, user_input):
        retrieved = self.retrieve_relevant_documents(user_input)
        return self.chain(user_input=user_input, context=retrieved)

    '''
    Method for getting the answer to a user's question along with the relevant context

    Parameters:
    - user_input: The user's input query/question which needs answering

    Returns:
    - The answer to the user's question along with the relevant context
    '''
    def get_answer_with_context(self, user_input):
        retrieved = self.retrieve_relevant_documents(user_input)
        return {"answer": self.chain(user_input=user_input, context=retrieved), "context": retrieved}

def main():
    rag_engine = RAGEngine()
    query = ""
    conversation_file = "rag-backend/conversation.txt"
    conversation_stack = deque()

    while query != "exit":
        query = input("Enter a query: \n")
        retrieved = rag_engine.retrieve_relevant_documents(query)
        chat_history = "\n".join(conversation_stack) if conversation_stack else ""
        response, generation_token_usage = rag_engine.chain(user_input=query, context=retrieved, chat_history=chat_history)
        input_tokens = generation_token_usage["input_tokens"]
        output_tokens = generation_token_usage["output_tokens"]
        total_tokens = generation_token_usage["total_tokens"]
        print(f"Input tokens: {input_tokens}, \tOutput tokens: {output_tokens}, \tTotal tokens: {total_tokens}")
        input_cost, output_cost, total_cost = rag_engine.calculate_generation_cost(input_tokens, output_tokens)
        
        with open(conversation_file, "a") as f:
            f.write(f"User: {query}\n")
            f.write(f"Assistant: {response}\n\n")
            f.write(f"Total cost: ${total_cost}\n\n")

        logger.info(f"Input tokens: {input_tokens}, \tInput cost: ${input_cost}")
        logger.info(f"Output tokens: {output_tokens}, \tOutput cost: ${output_cost}")
        logger.info(f"Generation cost: ${total_cost}")
        print(response)

        conversation_stack.append(query)
        conversation_stack.append(response)


if __name__ == "__main__":
    main()