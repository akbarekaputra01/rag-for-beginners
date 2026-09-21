import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# load_dotenv()

def load_documents(docs_path="docs"):
    """load all text files from the docs directory"""
    print(f"loading documents from {docs_path}...")
    
    # check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"the directory {docs_path} does not exist. please create it and add your company files")
    
    # load all .txt file from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise ValueError(f"no .txt files found in {docs_path}. please add your company documents")
    
    # show first 2 documents
    for i, doc in enumerate(documents[:2]):
        print(f"\ndocument {i+1}:")
        print(f"  source: {doc.metadata['source']}")
        print(f"  content length: {len(doc.page_content)} characters")
        print(f"  content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")
    
    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """split documents into smaller chunks with overlap"""
    print(f"splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- chunk {i+1}: ---")
            print(f"  source: {chunk.metadata['source']}")
            print(f"  length: {len(chunk.page_content)} characters")
            print(f"  content:")
            print(f"{chunk.page_content}")
            print("*" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """create and persist ChromaDB vector store"""
    print(f"creating embedding and storing in ChromaDB...")
    
    # embedding_model = OpenAIEmbeddings(
    #     model="text-embedding-3-small"
    # )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"--- creating vector store ---")
    vectorStore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print("--- finished creating vector store ---")
    
    print(f"vector store created and saved to {persist_directory}")
    
    return vectorStore

def main():
    print("Main Function")
    
    # loading the files
    documents = load_documents(docs_path="docs")
    
    # chunking the files
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=0)
    
    # embedding and storing in vector DB
    vector_store = create_vector_store(chunks, persist_directory="db/chroma_db")

if __name__ == "__main__":
    main()