import numpy as np
import openai
import pandas as pd
import os
import tiktoken
from scipy import spatial

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo-16k"

openai.api_key = os.getenv("OPENAI_API_KEY")

def search_similarity_strings(
        search_term: str,
        data_frame: pd.DataFrame,
        limit: int = 100
) -> tuple[list[str], list[float]]:
    """
    Search for strings in the data_frame that are most similar to the given search_term.
    
    Args:
    - search_term (str): The term to search for.
    - data_frame (pd.DataFrame): DataFrame containing strings and their embeddings.
    - limit (int): Maximum number of results to return.
    
    Returns:
    - tuple: Most similar strings and their similarity scores.
    """
    #Calculate the embedding for the search term
    embedding_response = openai.Embedding.create(model=EMBEDDING_MODEL, input=search_term)
    search_embedding = embedding_response["data"][0]["embedding"]

    #Define a function to calculate similarity between two embeddings
    def similarity(x,y):
        return 1 - spatial.distance.cosine(x,y)

    results = [
        (row["text"], similarity(search_embedding, row["embedding"]))
        for _, row in data_frame.iterrows()
    ]

    #sort result by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    #Extract texts and their scores
    texts, scores = zip(*results)

    return texts[:limit], scores[:limit]

def num_tokens(text: str, model: str = GPT_MODEL)->int:
    "return the number of token in a string"
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
) -> str:
    "Return a message for GPT with relevant source text pulled from a dataframe."
    strings, relatednesses = search_similarity_strings(query, df)
    print(strings)
    introduction = 'Use the below documentation on LangChain to answer the subsequent question. If the answer cannot be found in the documents, write "I could not find an answer.'
    question = f"\n\nQuestion: {query}"
    documentation = f"Here's the documentation {strings}"
    return message + question

def ask(
        query: str,
        df: pd.DataFrame = df,
        model: str = GPT_MODEL,
        print_message: bool = False,
) -> str:
    "Answers a query using GPT and a dataframe of relevant texts and embeddings."
    message = query_message_fixed(query,df, model=model)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about langchain"},
        {"role": "user", "content":message},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages = messages,
        temperature=0,
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

