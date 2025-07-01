

# ChromaRepository.py

## Purpose
- `ChromaRepository.py` provides functionality to interact with a Chroma collection, including adding, updating, deleting, and querying documents.

## Dependencies
- The file imports modules like `ChromaDb`, `Embeddings` from OpenAI and Ollama, `Document`, and others from various sources including many packages from `Langchain`

## Functionality
- The file contains functions for adding documents with or without metadata, updating documents, deleting documents, and querying the Chroma collection based on different criteria.

## Classes and Interfaces
- The file defines a class `ChromaRepository` that encapsulates the functionality related to interacting with the Chroma collection.
- the project also defines many retriever strategies using the following classes:
    - `SimilaritySearchRetriever`
    - `MultiSearchRetriever`
    - `SmallChunksSearchRetriever`

## Methods
- The class `ChromaRepository` contains methods like `add`, `update_by_id`, `delete_by_ids`, `get_all`, `getall_by_ids`, and more for managing the Chroma collection but also using the retrievers to gather data from the vector store.

## Testing
- Unit tests are available in `test_ChromaRepository.py` covering various functionalities of `ChromaRepository.py` which can be used to learn how to use the package.

## License
- The code is provided under the MIT License.