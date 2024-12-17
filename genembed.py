import openai
import os
import time
import logging
import chunks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 1000

def calculate_embeddings_for_dict(chunks_dict):
    """
    Calculate embeddings for a dictionary where each key is a file path and the corresponding value is a list of text chunks.
    
    Parameters:
    - chunks_dict (dict): Dictionary with file paths as keys and lists of text chunks as values.
    
    Returns:
    - Dictionary with file paths as keys and lists of embeddings as values.
    """
    

    embeddings_dict = {}

    for file_path, chunks in chunks_dict.items():
        embeddings = []

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch = chunks[batch_start:batch_end]
            print(f"Processing embeddings for {file_path}, batch {batch_start} to {batch_end-1}") 
            response = openai.Embedding.create(model = EMBEDDING_MODEL, input=batch)
            for i, be in enumerate(response["data"]):
                assert i == be["index"]
            batch_embeddings = [e["embedding"] for e in response["data"]]
            embeddings.extend(batch_embeddings)

        embeddings_dict[file_path] = embeddings

    return embeddings_dict

embeddings = calculate_embeddings_for_dict(results)


def generate_embeddings(text_chunks):
    #calculate embeddings for a list of text chunks

    embeddings = []
    total_chunks = len(text_chunks)

    for batch_start in range(0, total_chunks, BATCH_SIZE):
        batch = text_chunks[batch_start:batch_start + BATCH_SIZE]

        logging.info(f"Processing batch {batch_start//BATCH_SIZE + 1} out of {total_chunks//BATCH_SIZE + 1}")

        try:

            response = openai.Embedding.create(
                model = EMBEDDING_MODEL,
                input = batch
                )
            
            batch_embeddings = [data['embedding'] for data in response['data']]
            embeddings.extend(batch_embeddings)

        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            time.sleep(3)

        time.sleep(1)

    return embeddings

def process_directory_and_get_embeddings(directory, max_tokens=4096):

    """Process all HTML files within a directory (and its subdirectories), extracting text, 
    splitting based on token count, and then calculating embeddings for each chunk"""

    all_chunks = []
    file_paths = []

    for dirpath, dirname, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.html'):
                file_path = os.path.join(dirpath, filename)
                text = chunks.extract_text_from_html(file_path)
                chunk= chunks.split_text_by_tokens(text, max_tokens)
                all_chunks.extend(chunk)
                file_paths.extend([file_path] * len(chunks))

    #get embeddings for all chunks
    embeddings = generate_embeddings(all_chunks)

    #create a DataFrame with the file paths, chunks, and their embeddings
    df = pd.DataFrame({"file_path": file_paths, "text":all_chunks,"embedding":embeddings})

    return df


                