
# This code performs:
# - PDF Loading
# - Text Splitting
# - Text Embedding Generation
# - FAISS Vector Store Creation
# - Local Index Storage

# These libraries are imported to load PDF files, split documents into chunks, generate embeddings, and 
# store them into FAISS (Facebook AI Similarity Search).
# Boto3 is used to interact with AWS Bedrock services.

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import boto3


# boto3.client initializes the AWS Bedrock client to interact with Bedrock's runtime AP
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# BedrockEmbeddings generates text embeddings using Amazon Titan Embedding Model.
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Data Ingestion Function
def data_ingestion():
    # loader=PyPDFDirectoryLoader("../data/") # uncomment this if called from ingestion.py
    loader=PyPDFDirectoryLoader("data/") # use this if called from app.py
    documents=loader.load()
    print(f"Number of loaded documents: {len(documents)}")  # Debugging step
    
    # Document Splitting
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs=text_splitter.split_documents(documents)
    print(f"Number of split documents: {len(docs)}")
    
    return docs


# Vector store function:
# FAISS.from_documents() converts each document into embeddings using BedrockEmbeddings and stores them in a 
# FAISS vector store.
# save_local() saves the vector database locally into the faiss_index folder.
def get_vector_store(docs):
    vector_store_faiss=FAISS.from_documents(docs, bedrock_embeddings)
    vector_store_faiss.save_local("../faiss_index")
    return vector_store_faiss

    
if __name__ == '__main__':
    docs = data_ingestion()
    get_vector_store(docs)
    
    