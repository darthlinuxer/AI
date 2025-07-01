from chroma_repository.chroma_repository import ChromaRepository
from langchain_openai.embeddings import OpenAIEmbeddings

# This is a dummy test to check if the package can be imported and instantiated.
# It does not require an actual OpenAI API key or a persistent directory.
# If this runs without errors, the basic import and class instantiation are working.

try:
    # Using a placeholder for API key and directory for a basic import test
    # In a real scenario, you would provide valid credentials and paths.
    repo = ChromaRepository(
        persist_directory="./temp_chroma_db",
        collection_name="test_collection",
        embedding_function=OpenAIEmbeddings(api_key="sk-dummy-api-key"),
    )
    print("ChromaRepository imported and instantiated successfully!")
except Exception as e:
    print(f"Error importing or instantiating ChromaRepository: {e}")
