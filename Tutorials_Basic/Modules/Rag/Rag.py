from langchain_community.chat_models.ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain import hub
from langchain_core.retrievers import BaseRetriever

class Rag:

    EnableMultiQueryLog = True
    DatabaseDirectory = "./Tutorials_Basic/Databases"

    def __init__(
        self,
        llm_model: ChatOllama | ChatOpenAI,
    ):
        self._llm_model = llm_model        

    def talk_to_ai(self, text: str) -> AIMessage:
        return self._llm_model.invoke(text)

    def Run(
        self, question: str, retriever: BaseRetriever, stream: bool = False
    ) -> dict[str, str]:

        prompt: ChatPromptTemplate = hub.pull("rlm/rag-prompt")
        # https://smith.langchain.com/hub/rlm/rag-prompt

        def format_docs(docs: list[Document]) -> str:
            self.chunks = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            self.metadatas = [
                metadata
                for n, metadata in enumerate(metadatas)
                if metadata not in metadatas[n + 1 :]
            ]
            return "\n\n".join(self.chunks)

        def prepare_output_from_AIMessage(aiMessage: AIMessage) -> dict[str, str]:
            content = aiMessage.content
            metadata = aiMessage.response_metadata
            return {
                "content": content,
                "sources": self.metadatas,
                "chunks": self.chunks,
                "ai_metadata": metadata,
            }

        def inspect_the_prompt(prompt: ChatPromptTemplate) -> ChatPromptTemplate:
            from rich import print

            if self.EnableMultiQueryLog:
                print("Inspecting the prompt messages sent to the LLM:")
                if prompt.messages:
                    messages = [
                        f"{type(message).__name__}:{message.content}"
                        for message in prompt.messages
                    ]
                    for i, message in enumerate(messages):
                        print(f"{i}: {message}")
            return prompt

        chain = (
            RunnableParallel(
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | inspect_the_prompt
            | self._llm_model
            | prepare_output_from_AIMessage
        )

        if not stream:
            return chain.invoke(question)
        else:
            return chain.stream(question)
