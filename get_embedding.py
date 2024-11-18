from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function_aws():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        region_name="us-east-1"
    )
    return embeddings

def get_embedding_function_ollama():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings