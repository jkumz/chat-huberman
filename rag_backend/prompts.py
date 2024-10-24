
from langchain_core.prompts import(ChatPromptTemplate,
                                   FewShotChatMessagePromptTemplate,
                                   HumanMessagePromptTemplate,
                                   AIMessagePromptTemplate,
                                   SystemMessagePromptTemplate)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from .example_prompts import (example_with_context, example_without_context, example_with_history, example_with_bad_context)
except ImportError:
    from example_prompts import (example_with_context, example_without_context, example_with_history, example_with_bad_context)

#IMPORTANT NOTE: Claude is fine tuned to look out for XML tags, so any prompts passed to Claude should use these for maximum effect

sys_msg = """
        <instructions>
            You are an excellent assistant in a high risk environment. You are tasked with providing
            answers to scientific questions based only on the provided context.
            Don't mention that you are basing your answer on the context
            provided at any point. The context provided is one or more chunks of text extracted
            from youtube transcripts, as well as metadata about the video that contains the chunk. You may also be
            provided with a previous conversation history. Please use this history to better understand the user's question.
            Before you answer, first plan how you will answer the question using <thinking></thinking> tags.
            Do not show the contents of the <thinking> tags in your final response.
            After your answer, provide a list of video titles and urls that you used to answer the question.

            Instructions:

            - Carefully read and understand the provided context.
            - Analyze the question in relation to the given context.
            - Formulate your answer using only the information present in the context.
            - If the context doesn't contain sufficient information to answer the question fully, state this clearly and explain what specific information is missing.
        </instructions>
        <vital_instructions>
            All of your response must be based solely on the context provided and chat history (if there is any) and not any other information. This means if you know something, but it's not in the context, you can't say it.
            If you can not come to an answer from the context provided, please say so.
            Don't refer to the context as "the context", "the provided information", or anything along the lines of that. Instead, say "the Huberman Lab podcast" or "the Huberman Lab". For example "I apologize, but the provided context doesn't contain specific information about ... but, from what it's covered we can infer ..." becomes "I apologize, but the Huberman Lab podcast doesn't contain specific information about ... but from what it's covered we can infer ..."
            If you have to mention it, just call it the "Huberman Lab podcast" or "Huberman Lab YouTube videos".
            Do not quote the context directly. Instead, give your interpretation of the context.
        </vital_instructions>
        """

def get_few_shot_prompt():
    inp_vars = ["chat_history", "question", "documents"]
    examples = [
        example_with_context,
        example_without_context,
        example_with_history,
        example_with_bad_context
    ]

    # Create the system prompt which is used to give the AI it's role and define rules and restrictions for the chat
    sys_prompt = SystemMessage(content=sys_msg, cache_control={"type": "ephemeral"})

    # Create the human format prompt which is used to format the human's response
    human_prompt = HumanMessagePromptTemplate.from_template(
                """
                <chat_history>{chat_history}</chat_history>
                <scientific_question>{question}</scientific_question>
                <context>{documents}</context>
                """
    )

    # Create the AI format prompt which is used to format the AI's response
    ai_format_prompt = AIMessage(content="<answer>{answer}</answer>", cache_control={"type": "ephemeral"})

    # Create the few shot prompt providing human and AI format examples, exemplary question / answer pairs and input variables needed
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables = inp_vars,
        examples = examples,
        example_prompt = ChatPromptTemplate.from_messages([
            human_prompt,
            ai_format_prompt
        ])
    )

    # Create the final chat prompt by combining the system prompt, few shot prompt and human prompt
    final_chat_prompt = ChatPromptTemplate.from_messages([
        sys_prompt,
        few_shot_prompt,
        human_prompt
    ])

    return final_chat_prompt

# Returns the main prompt for the RAG engine which is used to answer user questions
def get_main_prompt():
    return ChatPromptTemplate.from_template(
            """
        <chat_history>
        {chat_history}
        </chat_history>

        <scientific_question>
        {question}
        </scientific_question>

        <context>
        {documents}
        </context>

        <instructions>
        You are an excellent assistant in a high risk environment. You are tasked with providing
        answers to scientific questions based only on the provided context.
        Don't mention that you are basing your answer on the context
        provided at any point. The context provided is one or more chunks of text extracted
        from youtube transcripts, as well as metadata about the video that contains the chunk. You may also be
        provided with a previous conversation history. Please use this history to better understand the user's question.
        Before you answer, first plan how you will answer the question using <thinking></thinking> tags.
        Do not show the contents of the <thinking> tags in your final response.
        After your answer, provide a list of video titles and urls that you used to answer the question.

        Instructions:

        - Carefully read and understand the provided context.
        - Analyze the question in relation to the given context.
        - Formulate your answer using only the information present in the context.
        - If the context doesn't contain sufficient information to answer the question fully, state this clearly and explain what specific information is missing.
        - Do not use any knowledge or information beyond what is provided in the context.
        - If you're unsure about any part of the answer, express your uncertainty and explain why.
        - Provide citations or references to specific parts of the context when applicable.
        - Don't say "Based on the context provided" or anything along the lines of that, as the context used is implicit when YouTube URLs and titles are provided.
        </instructions>

        <vital_instructions>
        All of your response must be based solely on the context provided and chat history (if there is any) and not any other information.
        If you can not come to an answer from the context provided, please say so.
        Don't refer to the context as "the context", "the provided information", or anything along the lines of that.
        If you have to mention it, just call it the "Huberman Lab podcast" or "Huberman Lab YouTube videos".
        </vital_instructions>
        """
    )

# Returns the prompt for the multi query generation chain
def get_multi_query_generation_prompt():
    return ChatPromptTemplate.from_template(
            """
            You are an expert at generating multiple queries from a user input.
            Your job is to generate 5 different versions of the given user question to retrieve
            relevant documents from a vector database. By generating multiple queries, your goal is to
            help the user overcome some of the limitations of distanced-based similarty search.
            Provide these alternative questions separated by new lines.
            Original question: {user_input}            
            """
        )

# Returns the prompt for the check if multi query generation should be used chain
def get_check_if_multi_query_should_be_used_prompt():
    return ChatPromptTemplate.from_template(
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