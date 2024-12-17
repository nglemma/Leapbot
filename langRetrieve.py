from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
#from langchain.chains import ConversationalRetrievalChain
#from langchain.chains.conversational_retrieval import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import langVecEmbeddings
from fastapi import FastAPI
from langserve import add_routes

retriever = langVecEmbeddings.vectorstore.as_retriever()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(),
    chain_type = "map_reduce",
    retriever=retriever,
    memory=memory
    )

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

#response = qa.run("how can students and recent graduates from the University of Bolton be supported")

#print(response)