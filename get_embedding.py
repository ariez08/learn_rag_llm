from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    # embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", credentials_profile_name="default", region_name="us-east-1")
    embeddings = OllamaEmbeddings(model="llama3.2")
    return embeddings