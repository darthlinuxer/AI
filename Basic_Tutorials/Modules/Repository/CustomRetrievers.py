from typing import Any, List, Optional
from abc import abstractmethod
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain.storage import InMemoryStore
from langchain.retrievers import MultiQueryRetriever
from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from scipy.spatial import distance
from hashlib import sha256
from .IVectorRepository import IVectorRepository


class ICustomRetrievers:
    @abstractmethod
    def fetch_documents(self, query: str) -> List[Any]:
        pass


class SimilaritySearchRetriever(VectorStoreRetriever, ICustomRetrievers):

    def __init__(
        self,
        vectorstore: VectorStore,
        max_number_of_documents: int,
        filter_by_metadata: dict[str, Any] = None,
        **kwargs
    ) -> None:
        """
        Initializes a SimilaritySearchRetriever object.

        Args:
            vectorstore (VectorStore): The vector store used for retrieval.
            max_number_of_documents (int): The maximum number of documents to retrieve.
            metadata (dict[str, Any], optional): Additional metadata for filtering. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        This function initializes a SimilaritySearchRetriever object by setting the search_kwargs based on the provided metadata.
        If metadata is provided, the search_kwargs are set to {"k": max_number_of_documents, "filter": metadata}.
        Otherwise, the search_kwargs are set to {"k": max_number_of_documents}.

        The function also sets the tags based on the provided kwargs. If "tags" is not present in kwargs, it defaults to None.

        Finally, the function calls the parent class's __init__ method with the provided vectorstore, search_type="similarity",
        search_kwargs, and tags.
        """
        search_kwargs = {}
        if filter_by_metadata:
            search_kwargs = {"k": max_number_of_documents, "filter": filter_by_metadata}
        else:
            search_kwargs = {"k": max_number_of_documents}

        tags = kwargs.pop("tags", None) or []
        super().__init__(
            vectorstore=vectorstore,
            search_type="similarity",
            search_kwargs=search_kwargs,
            tags=tags,
        )

    def fetch_documents(self, query: str) -> List[Any]:
        return self.invoke(query)


class MultiSearchRetriever(MultiQueryRetriever, ICustomRetrievers):

    @classmethod
    def from_llm(
        self,
        vectorstore: VectorStore,
        max_number_of_documents: int,
        llm: ChatOpenAI | ChatOllama,
        metadata: dict[str, Any] = None,
        include_original_question=True,
        prompt: PromptTemplate = None,
    ) -> MultiQueryRetriever:
        """
        Generates a Multiple Questions from a given context.
        if a prompt is provided it has to follow this template:
        (this is the default prompt template)
        - PromptTemplate(
                input_variables=["question"],
                template="You are an AI language model assistant. Your task is
                to generate 3 different versions of the given user
                question to retrieve relevant documents from a vector  database.
                By generating multiple perspectives on the user question,
                your goal is to help the user overcome some of the limitations
                of distance-based similarity search. Provide these alternative
                questions separated by newlines. Original question: {question}",

        Parameters:
            vectorstore: VectorStore - The vector store used for retrieval.
            max_number_of_documents: int - The maximum number of documents to retrieve.
            llm: ChatOpenAI | ChatOllama - The LLAMA model to use for retrieval.
            metadata: dict[str, Any] (optional) - Additional metadata for filtering.
            include_original_question: bool - Flag indicating whether to include the original question.
            prompt: PromptTemplate - The prompt template to use.

        Returns:
            MultiQueryRetriever - The generated MultiQueryRetriever based on the LLAMA model.
        """

        simple_retriever = SimilaritySearchRetriever(
            vectorstore=vectorstore,
            max_number_of_documents=max_number_of_documents,
            filter_by_metadata=metadata,
        )

        if not prompt:
            return super().from_llm(
                retriever=simple_retriever,
                llm=llm,
                include_original=include_original_question,
            )
        return super().from_llm(
            retriever=simple_retriever,
            llm=llm,
            prompt=prompt,
            include_original=include_original_question,
        )

    def fetch_documents(self, query: str) -> List[Any]:
        return self.invoke(query)


class SmallChunksSearchRetriever(BaseRetriever, ICustomRetrievers):

    vectorRepository: Optional[IVectorRepository]
    """The vectorrepository to use for the source documents"""
    filter_database: Optional[dict[str,Any]]
    """The metadatas to filter the vector database"""
    documents: Optional[list[Document]]
    """The documents to use for the source documents"""
    child_splitter: TextSplitter
    """The text splitter to use for the child documents"""
    embedding_function: Embeddings
    """The embedding function to use for the child documents"""
    byte_store: InMemoryStore
    """The lower-level backing storage layer for the child documents"""
    max_number_of_documents: int = 10
    """The maximum number of documents to retrieve"""
    max_number_of_fetched_documents_from_vector_store: int = 10

    _calculated_values: list[tuple] = None
    """internal calculated values"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        if self.vectorRepository is None and self.documents is None:
            raise ValueError(
                "Either vectorstore or documents must be provided to the SmallChunksSearchRetriever"
            )

        if self.child_splitter is None:
            raise ValueError(
                "The child_splitter must be provided to the SmallChunksSearchRetriever"
            )

        if self.embedding_function is None:
            raise ValueError(
                "The embedding_function must be provided to the SmallChunksSearchRetriever"
            )

        if self.byte_store is None:
            raise ValueError(
                "The byte_store must be provided to the SmallChunksSearchRetriever"
            )

        if self.documents is None and self.vectorRepository is not None:
            if self.filter_database:
                self.documents = self.vectorRepository.context_search_by_similarity_with_score(
                    query, k=self.max_number_of_fetched_documents_from_vector_store,
                    metadata_filter=self.filter_database
                )
            else:
                self.documents = self.vectorRepository.context_search_by_similarity_with_score(
                    query, k=self.max_number_of_fetched_documents_from_vector_store
                )
            #extract the document from the tuple
            self.documents = [document[0] for document in self.documents]

        parent_ids = []
        for document in self.documents:
            parent_id = sha256(document.page_content.encode("utf-8")).hexdigest()
            parent_ids.append(parent_id)
            sub_docs: List[Document] = self.child_splitter.split_documents([document])
            self._add_subdocs_to_memory(parent_id, sub_docs)

        sorted_calculated_sub_docs = self._calculate_subdocs_distance(query)

        unique_parent_doc_ids = []
        for calculated_sub_doc in sorted_calculated_sub_docs:
            if calculated_sub_doc[1] not in unique_parent_doc_ids:
                parent_id = calculated_sub_doc[1]  # parent id
                unique_parent_doc_ids.append(parent_id)

        limited_unique_parent_doc_ids = unique_parent_doc_ids[
            : self.max_number_of_documents
        ]

        build_reply = [
            self.documents[parent_ids.index(limited_unique_parent_doc_id)]
            for limited_unique_parent_doc_id in limited_unique_parent_doc_ids
        ]

        return build_reply

    def _add_subdocs_to_memory(self, parent_id:str, sub_docs:List[Document]):
        for subdoc in sub_docs:
            child_id = sha256(subdoc.page_content.encode("utf-8")).hexdigest()
            in_memory_data = self.byte_store.mget([child_id])
            if in_memory_data[0] is None:
                self.byte_store.mset(
                    [
                        (
                            child_id,
                            (
                                self.embedding_function.embed_query(
                                    subdoc.page_content
                                ),
                                parent_id,
                                subdoc.page_content,
                            ),
                        )
                    ]
                )
                
    def _calculate_subdocs_distance(self, query: str) -> List[tuple]:
        """calculate the distance of the query string
        in the relation to each embedding_content in the database
        """
        embedded_query = self.embedding_function.embed_query(query)
        calculated_sub_docs = []
        for key in self.byte_store.yield_keys():
            vector = self.byte_store.mget([key])[0]
            if vector is not None:
                embedding_content = vector[0]  # embedding vector
                parent_id = vector[1]  # parent id
                subdoc = vector[2]  # subdoc content
                # calculate the distance of the query vector
                # in the relation to each embedding_content in the database
                calculated_distance = distance.euclidean(
                    embedded_query, embedding_content
                )
                calculated_sub_docs.append(
                    (calculated_distance, parent_id, subdoc)
                )  # append a tuple

        sorted_calculated_sub_docs = sorted(
            calculated_sub_docs, key=lambda x: x[0]
        )  # sort by calculated_distance, ascending order

        return sorted_calculated_sub_docs

    def fetch_documents(self, query: str) -> List[Any]:
        return self._calculate_subdocs_distance(query)
