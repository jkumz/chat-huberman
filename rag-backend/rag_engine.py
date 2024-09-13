from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv, find_dotenv

import os
import tiktoken


load_dotenv(find_dotenv(filename=".rag_engine.env"))

claude_sonnet_model="claude-3-5-sonnet-20240620" # most powerful, $3 per 1m tokens in, $15 per 1m tokens out
claude_haiku_model="claude-3-haiku-20240307" # less powerful, $0.25 per 1m tokens in, $1.25 per 1m tokens out
claude_opus_model="claude-3-opus-20240229" # most performant, $15 per 1m tokens in, $75 per 1m tokens out

#TODO: Calculate cost per query after the fact, display to user
#TODO: Memory: Remember previous questions and answers in chat, and use them to inform the current answer
#TODO: Ability to go back to previous chats - MAYBE - this would require storing it in a database with user id etc

# Something to consider: Should we make a Base Class, and then other RAG engines inherit from it using different LLMs
class RAGEngine:
    '''
    Constructor class for the RAG engine class
    
    Parameters:
    - index_name: The name of the Pinecone index to use
    - llm: The LLM to use for generating responses
    - embedding_model: The embedding model to use for generating embeddings
    - output_parser: The output parser to use for parsing the output of the LLM
    '''
    def __init__(self, model=claude_sonnet_model):
        self.embedding_model = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
        self.index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(name=os.getenv("INDEX_NAME"), host=os.getenv("INDEX_HOST"))
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self.set_model(model)
        self.output_parser = StrOutputParser()

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
        if self.model == claude_sonnet_model:
            self.input_cost_per_token = 0.000003  # $3 per 1M tokens
            self.output_cost_per_token = 0.000015  # $15 per 1M tokens
        elif self.model == claude_haiku_model:
            self.input_cost_per_token = 0.00000025  # $0.25 per 1M tokens
            self.output_cost_per_token = 0.00000125  # $1.25 per 1M tokens
        elif self.model == claude_opus_model:
            self.input_cost_per_token = 0.000015  # $15 per 1M tokens
            self.output_cost_per_token = 0.000075  # $75 per 1M tokens
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
    def calculate_cost(self, input_text, output_text):
        input_tokens = len(self.tokenizer.encode(input_text))
        output_tokens = len(self.tokenizer.encode(output_text))
        input_cost = input_tokens * self.input_cost_per_token
        output_cost = output_tokens * self.output_cost_per_token
        total_cost = input_cost + output_cost
        return input_cost, output_cost, total_cost, input_tokens, output_tokens

    '''
    Method for chaining together the components of the RAG engine
    
    Parameters:
    - user_input: The user's input query/question which needs answering
    - context: The context to use for generating the response
    
    Returns:
    - The response from the LLM
    '''
    # TODO: Check answer loop: Use ChatGPT to check if the answer is correct or "answered"
    # if it isn't, then we need to refine the question and re-run the chain and try fill in the blanks.
        
    # TODO: We can also look through the context, and if there is context that is not fully completed, we can
    # generate a query that will retrieve more relevant context and add it to the context list
    # Example: "Implementing the big 6 pillars of health: <6 pillars not mentioned, so LLM defines
    # a query that will retrieve more context on the 6 pillars and add it to the context list>"

        # TODO: Few shot learning: Give examples of previous conversations to the LLM to help it answer the question
        # - No "Based on the context provided" or "The context provided is" in the answer
    # - No "I used the context provided to answer your question" in the answer
    # - Sound like a human, not an AI
    def chain(self, user_input, context):
        prompt = ChatPromptTemplate.from_template(
            """
        You are an excellent assistant in a high risk environment. You are tasked with providing
        answers to scientific questions based only on the provided context. If you can not come to an answer
        from the context provided, please say so. The context provided is one or more chunks of text extracted
        from youtube transcripts, as well as metadata about the video that contains the chunk. After your answer,
        provide a list of video titles and urls that you used to answer the question.

        Instructions:

        - Carefully read and understand the provided context.
        - Analyze the question in relation to the given context.
        - Formulate your answer using only the information present in the context.
        - If the context doesn't contain sufficient information to answer the question fully, state this clearly and explain what specific information is missing.
        - Do not use any knowledge or information beyond what is provided in the context.
        - If you're unsure about any part of the answer, express your uncertainty and explain why.
        - Provide citations or references to specific parts of the context when applicable.
        - Do not tell the user that you are basing your answer on any context, just answer the question.

        The question you must answer is: {question}

        The provided context is: {documents}
        """)

        chain = prompt | self.llm | self.output_parser
        return chain.invoke({"question": user_input, "documents": context})
    
    '''
    Method for retrieving the most relevant chunks from the index
    
    Parameters:
    - user_input: The user's input query/question which needs answering
    
    Returns:
    - The most relevant chunks from the index
    ''' 
    # Most of the following ideas would be using a cheaper LLM to save costs as Claude 3.5 Sonnet is very expensive
    #TODO: Query transformations/translations
    #TODO: RAG Fusion Reranking
    #TODO: Query decompostion: If input is a complex problem, break it down into sub-queries - how do we gauge complexity?
    #TODO: MAYBE: Step back approach - Make question more abstract, and then step by step make it more specific until we have a relevant context
    #TODO: HyDE: Generate a hypothetical "document" that would answer the question, embed it, then use that to get relevant context.
    #      So to clarify, we can't answer it without context, but we can say what good context would look like, and then go find it.
    def retrieve_relevant_chunks(self, user_input):
        #TODO: Use metadata of chunks as well as content to determine relevance,
        # for example if the title of the video is semantically similar to the user input,
        # then we can go more in depth on that video
        return self.retriever.invoke(user_input)

def main():
    # test retrieval
    rag_engine = RAGEngine()
    query = ""
    conversation_file = "rag-backend/conversation.txt"
    
    while query != "exit":
        query = input("Enter a query: ")
        
        retrieved = [{"content": doc.page_content, "metadata": doc.metadata} for doc in rag_engine.retrieve_relevant_chunks(query)]
        response = rag_engine.chain(user_input=query, context=retrieved)
        
        with open(conversation_file, "a") as f:
            f.write(f"User: {query}\n")
            f.write(f"Assistant: {response}\n\n")

        input_text = query + ''.join([f"{doc['content']}{doc['metadata']}" for doc in retrieved])
        input_cost, output_cost, total_cost, input_tokens, output_tokens = rag_engine.calculate_cost(input_text, response)
        print(f"Input cost: ${input_cost}, Output cost: ${output_cost}, Total cost: ${total_cost}")
        print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        
        print(response)

if __name__ == "__main__":
    main()