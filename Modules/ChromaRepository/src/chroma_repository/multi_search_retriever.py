from langchain.retrievers import MultiQueryRetriever
from langchain_core.vectorstores import VectorStore
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from typing import Any, List
from .icustom_retriever import ICustomRetriever
from .similarity_search_retriever import SimilaritySearchRetriever

class MultiSearchRetriever(MultiQueryRetriever, ICustomRetriever):

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
