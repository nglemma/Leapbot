import json
import openai
#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from tqdm import tqdm
import os

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
model_name = 'text-embedding-ada-002'
#'text-embedding-ada-002'
#list to store loaded documents
documents = []

#Load documents from a JSon lines file
with open('/Users/mac/Documents/Chatbot/train_2.jsonl', 'r') as file:
    for line in file:
        #parse each line as a JSON object and append to the 'documents' list
        documents.append(json.loads(line))

embeddings = OpenAIEmbeddings(model = model_name)

vectorstore = Chroma(
    collection_name = "langchain_store",
    embedding_function = embeddings,
    persist_directory ="./chroma_db"
)

for doc in tqdm(documents):
    vectorstore.add_texts(texts=[str(doc)])

#query = "do you support students and recent graduates from the University of Bolton"
#docs = vectorstore.similarity_search(query)
#print(docs[0])




