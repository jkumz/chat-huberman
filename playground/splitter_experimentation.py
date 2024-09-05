from langchain_ai21 import AI21SemanticTextSplitter
import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

semantic_text_splitter = AI21SemanticTextSplitter()
strings_to_split = []
split_strings = []

num_chars = 0
with open("transcripts/transcription1.txt", "r") as file:
    while True:
        text_to_split = file.read(99999)
        if not text_to_split:
            break
        strings_to_split.append(text_to_split)
        num_chars += len(text_to_split)

for string in strings_to_split:
    if len(string) < 30:
        continue
    docs = semantic_text_splitter.split_text(string)
    if len(docs) != 0:
        split_strings.append(docs)

print(f"There is {len(split_strings[0]) + len(split_strings[1])} semantic splits from {num_chars} characters\n")
print(f"The mean split length is {(int)(num_chars/(len(split_strings[0]) + len(split_strings[1])))}")

with open("semantic-splitter-text.txt", "a", encoding="utf-8") as splitter_file:
    count = 0
    for split_list in split_strings:
        for split in split_list:
            splitter_file.write(f"Split {count+1}: {split}\n\n")
            count += 1

embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma("Prototype-Vector", embedding_model)
    
for large_chunk in split_strings:
    vectorstore.add_texts(large_chunk)

retriever = vectorstore.as_retriever()

template = """Answer the following question based only on the following context:
{context}

Question: {question}"""

prompt = ChatPromptTemplate.from_template(template=template)

question = ""
llm = ChatOpenAI(model="gpt-4o", temperature=0)

while question != "quit":
    print("enter question:\n")
    question = input()
    context = retriever.invoke(question)
    chain = prompt | llm
    print(chain.invoke({"context":context, "question":question}).content)



