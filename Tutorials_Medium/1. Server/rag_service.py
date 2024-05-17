from ChromaRepository import ChromaRepository, IVectorRepository

class RagService:
    
    @classmethod
    def setUpClass(cls):
        try:
            cls.vectorrepository: IVectorRepository = ChromaRepository(
                persist_directory=str("./database"),
                collection_name="youtube",
                embedding_function=OpenAIEmbeddings(
                    api_key=OPENAI_API_KEY,
                ),
            )

        except Exception as e:
            print(f"Error setting up class: {e}")