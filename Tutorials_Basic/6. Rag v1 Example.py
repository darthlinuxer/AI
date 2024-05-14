from Modules.Rag import Rag, SuperSourceLoaderTransformer
from Modules.Repository import ChromaRepository, SmallChunksSearchRetriever
from rich import print
from langchain_core.messages.ai import AIMessage
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json

os.system("cls" if os.name == "nt" else "clear")

from dotenv import load_dotenv

load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = SuperSourceLoaderTransformer()

llm_model = None
embed_model = None

llm_to_use = (
    input("Do you want to use OpenAI or Ollama LLM ? (default: OpenAI) -> ") or "OpenAI"
)
embedding_to_use = (
    input("Which embedding functon ? Ollama or OpenAI ? (default: Ollama) -> ")
    or "Ollama"
)
if llm_to_use == "OpenAI":
    llm_model = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
else:
    llm_model = ChatOllama(model="phi3")

# YOU HAVE TO USE THE SAME EMBEDDING FUNCTION THAT WAS USED TO ADD THE DOCUMENTS TO THE VECTOR STORE
if embedding_to_use == "Ollama":
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
else:
    embed_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

vectorRepository = ChromaRepository(
    persist_directory="./Tutorials_Basic/Databases",
    collection_name="personal",
    embedding_function=embed_model,
)
rag = Rag(llm_model=llm_model)
storage = InMemoryStore()
retriever = SmallChunksSearchRetriever(
    vectorRepository=vectorRepository,
    embedding_function=embed_model,
    child_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    ),
    byte_store=storage,
    max_number_of_documents=3,
)


def pause():
    input("Press Enter to continue...")


def loadDataFromSource(source: str, **kwargs) -> tuple[str, dict]:
    transcript = ""
    if "language" in kwargs:
        transcript = loader.loadDataFromSource(source, language=kwargs["language"])
    else:
        transcript = loader.loadDataFromSource(source)
    print(transcript)
    content = transcript[0].page_content
    is_punctuated = input("Is the transcript punctuated ? (y/n) : ")
    if is_punctuated == "n":
        print("Punctuating the transcript...")
        print("Punctuation is necessary for semantic chunking...")
        prompt = f"""Given the following context:
        ==========================
        Context: {content}
        ==========================
        Action:
        Perform the necessary punctuations and correct the grammar according to the provided language and 
        return the answer in the same format as the context.
        """
        aiReply: AIMessage = rag.talk_to_ai(prompt)
        content = aiReply.content
    return (content, transcript[0].metadata)


def semantic_chunk_text(content) -> list[Document]:
    print("Splitting the content into chunks...")
    while True:
        threshold = input(f"Enter the threshold : default 60 -> ") or 60
        chunks = loader.semantic_chunk_text(
            content, threshold=int(threshold), embed_model=embed_model
        )
        return chunks


def semantic_chunk_text_with_similar_token_size(content) -> list[Document]:
    print("Splitting the content into chunks...")
    while True:
        token_size = input(f"what is the token size? default 500 -> ") or 500
        # convert token size to int
        token_size = int(token_size)
        chunks = loader.semantic_chunk_with_token_size(content, chunk_size=token_size)
        result = [
            f"tokens: {loader.calculate_tokens([chunk])} -> {chunk}" for chunk in chunks
        ]
        print(result)
        return chunks


def add_to_chroma(chunks: list[str], metadata: dict) -> list[str]:
    ids = loader.create_hashids_for_texts(chunks)
    print("Current metadata : ", metadata)
    while True:
        add_custom_metadata = input(
            "Do you want to add any extra custom metadata ? (y/n) : "
        )
        if add_custom_metadata == "y":
            key = input("Enter custom key : ")
            value = input("Enter custom value: ")
            metadata[key] = value
            print(metadata)
        else:
            break
    print("Adding to chroma...")
    metadatas = [metadata] * len(chunks)
    return vectorRepository.add(texts=chunks, metadatas=metadatas, ids=ids)


content: str = ""
chunks: list[str] = []
filter: dict = {}


menu = """
0. Exit
1. load from a new source
2. split into chunks semantically
3. Split into chunks with similar token count
4. add chunks to chroma collection
5. Talk to your collection
6. clean chroma collection
7. print loaded metadata
8. print loaded content
9. print loaded chunks
10. clear all variables
11. Send content to AI and get response
12. Print content token count
13. Print chunks token count
14. Add/update metadata
15. Add/Update content
16. get all chunks from vectordb using metadatas
17. get all chunks from vectordb using similarity with score
18. get all chunks from vectordb 
"""

while True:
    os.system("cls" if os.name == "nt" else "clear")
    print(menu)
    choice = input("Enter your choice: ")
    if choice == "0":
        break
    if choice == "1":
        source = input("Type the source: ")
        try:
            if source == "":
                raise ValueError("Source should be a PDF, DOC, DOCX, or http link")
            else:
                language = (
                    input(
                        "if the source is a youtube link, enter the language: (en) -> "
                    )
                    or "en"
                )
                content, filter = loadDataFromSource(source, language=language)
        except Exception as e:
            print(e)

    elif choice == "2":
        chunks = semantic_chunk_text(content)
        print(chunks)
    elif choice == "3":
        chunks = semantic_chunk_text_with_similar_token_size(content)
    elif choice == "4":
        try:
            if chunks is None or chunks == []:
                print("Please prepare the chunks first")
            else:
                result = add_to_chroma(chunks, filter)
                print(result)
        except Exception as e:
            print(e)
    elif choice == "5":
        try:
            question = input("Question: ")
            filter = input("provide a filter metadata if any: ") or None
            if filter is not None:
                filter = json.loads(filter)  # this converts into a dictionary
            if filter:
                retriever = SmallChunksSearchRetriever(
                    vectorRepository=vectorRepository,
                    embedding_function=embed_model,
                    child_splitter=RecursiveCharacterTextSplitter(
                        chunk_size=200,
                        chunk_overlap=20,
                    ),
                    byte_store=storage,
                    max_number_of_documents=3,
                    filter_database=filter,
                )
            print(f"Answering the question: {question}")
            for chunk in rag.Run(question=question, retriever=retriever, stream=True):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(e)
    elif choice == "6":
        vectorRepository.delete_all()
        print("Collection deleted successfully")
    elif choice == "7":
        print(filter)
    elif choice == "8":
        print(content)
    elif choice == "9":
        print(chunks)
    elif choice == "10":
        content = ""
        chunks = []
        filter = {}
    elif choice == "11":
        if content != "":
            prompt_template = """Given the following context: 
            ========================
            {content}
            ========================
            Instructions: {command}
            """
            parsed_prompt = prompt_template.format(
                content=content, command=input("Type Command: ")
            )
            print("Parsed prompt sent to the LLM:")
            print(parsed_prompt)
            result = rag.talk_to_ai(parsed_prompt)
            print(f"content before transformation: {content}")
            print({"content": result.content, "metadata": result.response_metadata})
            accept_new_content = input("Accept new content ? (y/n) : ")
            if accept_new_content == "y":
                content = result.content
        else:
            print("No content to send to AI")

    elif choice == "12":
        print("Calculating the number of tokens in the content...")
        print(loader.calculate_tokens([content]))
    elif choice == "13":
        print("Calculating the number of tokens in the chunks...")
        print(loader.calculate_tokens(chunks))
    elif choice == "14":
        updated_metadata = input("Enter the metadata: ")
        if filter is not None:
            new_metadata = json.loads(updated_metadata)
            for key, value in new_metadata.items():
                filter[key] = value

            print("Metadata updated successfully.")
            print(filter)
        else:
            print("Invalid metadata. Please try again.")
    elif choice == "15":
        updated_content = input("Enter the content: ")
        if content is not None:
            content = updated_content
            print("Content updated successfully.")
            print(content)
    elif choice == "16":
        try:
            limit = int(input("Enter the number of chunks to retrieve : ")) or 1
            metadata = input("Enter the metadata : ")
            result = vectorRepository.getall_by_metadata(
                limit=limit, metadata_query=json.loads(metadata)
            )
            print(result)
        except Exception as e:
            print(e)
    elif choice == "17":
        try:
            limit = int(input("Enter the number of chunks to retrieve : ")) or 1
            context = input("Enter the context to search : ")
            result = vectorRepository.context_search_by_similarity_with_score(
                context=context, k=limit
            )
            print(result)
        except Exception as e:
            print(e)
    elif choice == "18":
        try:
            limit = int(input("Enter the number of chunks to retrieve : ")) or 1
            result = vectorRepository.get_all(limit=limit)
            print(result)
        except Exception as e:
            print(e)
    else:
        print("Invalid choice. Please try again.")

    pause()
