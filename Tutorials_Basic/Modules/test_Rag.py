from dotenv import load_dotenv
import os

load_dotenv("../../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import unittest
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Repository import ChromaRepository, IVectorRepository, SmallChunksSearchRetriever
from Rag.SuperSourceLoaderTransformer import SuperSourceLoaderTransformer
from Rag.Rag import Rag


class TestRagMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.vectorrepository: IVectorRepository = ChromaRepository(
                persist_directory="./Tutorials_Basic/Databases",
                collection_name="youtube",
                embedding_function=OpenAIEmbeddings(
                    api_key=OPENAI_API_KEY,
                ),
            )

            cls.rag = Rag(ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY))

            cls.superloader = SuperSourceLoaderTransformer()

            doc: list[Document] = cls.superloader.loadDataFromSource(
                "https://www.youtube.com/watch?v=9L77QExPmI0"
            )

            texts = doc[0].page_content
            metadata = doc[0].metadata
            metadata["source"] = "youtube"

            # save texts and metadata to a file
            with open("./Tutorials_Basic/Output/youtube.txt", "w") as f:
                f.write(texts)
                f.write("\n")
                f.write(metadata)

            # check OpenAI model context sizes here:
            # https://platform.openai.com/docs/models/gpt-3-5-turbo
            chunks: list[str] = cls.superloader.semantic_chunk_with_token_size(
                texts, chunk_size=8000
            )
            # just for verification purposes if the sizes are less than specified above
            chunk_token_sizes = cls.superloader.calculate_tokens(chunks)

            from hashlib import sha256

            metadatas = [metadata] * len(chunks)
            ids = [sha256(text.encode("utf-8")).hexdigest() for text in chunks]

            added_ids = cls.vectorrepository.add(
                texts=chunks, metadatas=metadatas, ids=ids
            )

        except Exception as e:
            print(e)
            pass

    def setUp(self):
        pass

    def tearDown(self):
        # Clean up any resources after each test method
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_rag_talk_to_ai(self):
        try:
            question = "summarize the context"
            context = self.vectorrepository.getall_by_metadata(
                {"source": "youtube"}, limit=3
            )

            texts = [doc for doc in context["documents"]]

            answer: AIMessage = self.rag.talk_to_ai(
                f"""
                Based on the context below: 
                ======================================
                Context: {texts}
                ======================================
                Question: {question}
                """,
            )

            self.assertIsNotNone(texts)
            self.assertIsNotNone(answer.content)
        except Exception as e:
            print(e)
            pass

    def test_rag_using_retriever_strategy_with_vectorrepository(self):
        """
        A method to test the RAG using a retriever. It initializes the retriever,
        runs a question through the RAG, fetches documents, and performs assertions on
        the results.
        """
        try:
            byte_store = InMemoryStore()
            # """to store the chunks processed by the retriever strategy"""

            # initialize the retriever
            # SmallChunkSearch retrievers are useful when your vectorstore contains
            # large chunks of text > 4000 tokens and you ask a question which
            # have more similarity to the chunks of the context than the context itself
            retriever = SmallChunksSearchRetriever(
                vectorRepository=self.vectorrepository,
                embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
                child_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=200, chunk_overlap=50
                ),
                byte_store=byte_store,
                max_number_of_documents=1,
                max_number_of_fetched_documents_from_vector_store=10,
            )
            # """ SmallChunksSearchRetriever strategy using vectorstore
            #     search you query in all vectordatabase and return contexts by similarity
            #     up to the max_number_of_fetched_documents_from_vector_store
            #     split the context in small parts (chunks) and calculate the euclidean vector
            #     distance between the query and the chunks
            #     save the chunks in a byte store in case you want to make another question later
            #     with the calculated distance,sort the chunks
            #     from the top sorted splits, fetch the parent id of the main context used
            #     return the parent document up to the max_number_of_documents
            # """

            question = "how to create a logger in python ?"
            result = self.rag.Run(question, retriever, stream=False)
            # using the context retrieved by the retriever strategy, feed the LLM with
            # the question

            processed_chunks = retriever.fetch_documents(question)
            # retrieve all the processed chunks from the InMemoryStorage with the chunks
            # sorted by distance

            self.assertIsNotNone(processed_chunks)
            self.assertIsNotNone(result["content"])
            self.assertIsNotNone(result["chunks"])
            self.assertEqual(result["sources"]["source"], "youtube")

        except Exception as e:
            print(e)

    def test_rag_using_retriever_strategy_with_documents(self):
        """
        A method to test the RAG using a retriever. It initializes the retriever with
        provided documents, runs a question through the RAG, and performs assertions on
        the results based on the provided documents.
        """
        try:
            byte_store = InMemoryStore()
            # """to store the chunks processed by the retriever strategy"""

            contents = self.vectorrepository.get_all()
            documents = [
                Document(page_content=content) for content in contents["documents"]
            ]

            # initialize the retriever
            # SmallChunkSearch retrievers are useful when your vectorstore contains
            # large chunks of text > 4000 tokens and you ask a question which
            # have more similarity to the chunks of the context than the context itself
            retriever = SmallChunksSearchRetriever(
                documents=documents,
                embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
                child_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=200, chunk_overlap=50
                ),
                byte_store=byte_store,
                max_number_of_documents=1,
            )
            # """ SmallChunksSearchRetriever strategy using vectorstore
            #     search you query in the provided documents and return contexts by similarity
            #     up to the max_number_of_fetched_documents_from_vector_store
            #     split the context in small parts (chunks) and calculate the euclidean vector
            #     distance between the query and the chunks
            #     save the chunks in a byte store in case you want to make another question later
            #     with the calculated distance,sort the chunks
            #     from the top sorted splits, fetch the parent id of the main context used
            #     return the parent document up to the max_number_of_documents
            # """

            question = "how to create a logger in python ?"
            result = self.rag.Run(question, retriever, stream=False)
            # using the context retrieved by the retriever strategy, feed the LLM with
            # the question

            content = result["content"]
            chunks = result["chunks"]

            processed_chunks = retriever.fetch_documents(question)
            # retrieve all the processed chunks from the InMemoryStorage with the chunks
            # sorted by distance

            self.assertIsNotNone(processed_chunks)
            self.assertIsNotNone(content)
            self.assertIsNotNone(chunks)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    unittest.main(buffer=False)
