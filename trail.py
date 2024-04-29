import requests
import streamlit as st
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
#import chromadb
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import pprint
import os
import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# URL of the raw vector embeddings file on GitHub
url = 'https://github.com/naren579/EMPLOCHAT/raw/main/embeddings/db/chroma.sqlite3'

# Path where the file will be saved locally
local_path = 'app/vector_embeddings'

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(local_path), exist_ok=True)

# Download the file if it doesn't already exist
if not os.path.exists(local_path):
    with st.spinner('Downloading vector embeddings...'):
        response = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(response.content)

# Now you can load the vector embeddings file into your script
#Initialize the Chroma DB client
store = Chroma(persist_directory=local_path,collection_name="Capgemini_policy_embeddings")

# Get all embeddings
embeddings = store.get(include=['embeddings'])

######################Getting Similar Vector Embeddings for a given prompt#####

def retrieve_vector_db(query, n_results=3):
    similar_embeddings = store.similarity_search_by_vector_with_relevance_scores(embedding = embed_prompt.embed_query(query),k=n_results)
    results=[]
    prev_embedding = []
    for embedding in similar_embeddings:
      if embedding not in prev_embedding:
        results.append(embedding)

      prev_embedding =  embedding
    return results


# The loading code will depend on the format of your vector embeddings file

query = st.text_input("Enter your question:")
if st.button('Go'):
  st.write(retrieve_vector_db(query, n_results=3))
