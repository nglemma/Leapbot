from bs4 import BeautifulSoup
import tiktoken
import os
import logging
import genembed
from genembed import generate_embeddings
import pandas as pd

def extract_text_from_html(file_path):

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        text = soup.get_text()

        clean_text = ' '.join(text.split())

        return clean_text

    except Exception as e:
        return f"Error occured while processing the file:{str(e)}"
  
def split_text_by_tokens(text, max_tokens=300, encoding_name='gpt-3.5-turbo'):

    #this is the tokenizer
    encoding = tiktoken.encoding_for_model(encoding_name)

    words = text.split()

    #spliting text to chunks
    chunks = []
    current_chunk = ""

    for word in text.split():
        #check if adding the word does not exceed the max_token count
        if len(encoding.encode(current_chunk + " " + word)) <= max_tokens:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word
    #appending any remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def process_directory(directory, max_token=400):
   #processes a directory of html files, extratcs text, tokenizes and generates embeddings.
   
   results = {}
   for dirpath, dirnames, filenames in os.walk(directory):
      for filename in filenames:
         if filename.endswith('.html'):
            file_path = os.path.join(dirpath, filename)
            logging.info(f"processing file: {file_path}")

            text = extract_text_from_html(file_path)
            if not text:
               continue
            
            #tokenize text into chunks
            text_chunks = split_text_by_tokens(text, max_token)
            results[file_path] = text_chunks

   return results



