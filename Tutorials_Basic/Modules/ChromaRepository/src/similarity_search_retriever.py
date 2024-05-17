from typing import Any, List
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from .icustom_retriever import ICustomRetriever

class SimilaritySearchRetriever(VectorStoreRetriever, ICustomRetriever):

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
