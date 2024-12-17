from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
import tiktoken
import json

#loader = ReadTheDocsLoader('/Users/mac/Documents/Chatbot/Testdata')
loader = DirectoryLoader(
    '/Users/mac/Documents/Chatbot/Testdata/', 
    glob="*.txt",  # Adjust to the extension of your documents
    loader_cls=TextLoader
)

#load documents and calculate the number of documents
docs = loader.load()
print(f"Loaded {len(docs)} documents")

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    #Encode the text into tokens and count the number of tokens
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

#configure a text splitter that breaks up text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function = tiktoken_len,
    separators=['\n\n','\n',' ',' ']
)

#save the documents into a jsonl file
output_file = '/Users/mac/Documents/Chatbot/train_2.jsonl'
with open(output_file, 'w') as f:
    for doc in docs: 
        #convert the documents into a json string and write to the file with a newline
        doc_dict = {
            'content': doc.page_content,  # Assuming 'text' contains the document text
            'metadata': doc.metadata  # Assuming 'metadata' contains any metadata
        }
        f.write(json.dumps(doc_dict) + '\n')

print(f"Documents has been saved in {output_file}")