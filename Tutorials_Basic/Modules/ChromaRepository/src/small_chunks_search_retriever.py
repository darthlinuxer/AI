from typing import Any, List, Optional
from langchain.storage import InMemoryStore
from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from scipy.spatial import distance
from hashlib import sha256
from .ivector_repository import IVectorRepository
from .icustom_retriever import ICustomRetriever

class SmallChunksSearchRetriever(BaseRetriever, ICustomRetriever):

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
