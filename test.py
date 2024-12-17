import os
import chunks
from chunks import process_directory

test_directory = '/Users/mac/Documents/Chatbot/testhtml'
results = chunks.process_directory(test_directory, max_token=50)
for file, chunks in results.items():
    print(f"File: {file}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")