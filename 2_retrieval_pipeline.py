from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

persistent_directory = "db/chroma_db"

# load embedding and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# search for relevant documents
query = "siapa akbar?"

retriever = db.as_retriever(search_kwargs={"k": 3})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3 # only return chunks with cosine similarity >= 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"query: {query}")

print(f"--- context ---")
# display results
for i, doc in enumerate(relevant_docs):
    print(f"document {i}:\n{doc.page_content}\n")

# Initialize dengan custom base_url
llm = ChatOpenAI(
    model="kr/claude-sonnet-4.5",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.7
)

# ==================================================
# Define shared components (dipakai LCEL dan Manual)

def format_docs(docs):
    """Convert list of documents to single string"""
    return "\n\n".join([doc.page_content for doc in docs])

prompt_template = """Based on the following context, please answer the question.
If you don't know the answer, say "I don't have enough information to answer based on the provided documents."

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ==================================================
# OPTION 1: LCEL (LangChain Expression Language) - OTOMATIS

print("\n" + "="*50)
print("OPTION 1: LCEL Approach (Automatic)")
print("="*50)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# Test RAG chain
question = query
answer_lcel = rag_chain.invoke(question)

print(f"\n--- RAG Answer (LCEL) ---")
print(answer_lcel)

# ==================================================
# OPTION 2: MANUAL Approach - STEP BY STEP

print("\n" + "="*50)
print("OPTION 2: Manual Approach (Step-by-Step)")
print("="*50)

# Step 1: Retrieve documents
print("\n[Step 1] Retrieving documents...")
docs = retriever.invoke(query)
print(f"✓ Retrieved {len(docs)} documents")

# Step 2: Format documents jadi string
print("\n[Step 2] Formatting documents...")
formatted = format_docs(docs)
print(f"✓ Formatted into {len(formatted)} characters")

# Step 3: Format prompt dengan context dan question
print("\n[Step 3] Creating prompt...")
prompt_value = PROMPT.format(context=formatted, question=query)
print(f"✓ Prompt created ({len(prompt_value)} characters)")

# Step 4: Send ke LLM
print("\n[Step 4] Sending to LLM...")
response = llm.invoke(prompt_value)
print(f"✓ Got response from LLM")

# Step 5: Extract content
print("\n[Step 5] Extracting answer...")
answer_manual = response.content
print(f"✓ Answer extracted")

print(f"\n--- RAG Answer (Manual) ---")
print(answer_manual)

# ==================================================
# Comparison

print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"LCEL Answer Length: {len(answer_lcel)} characters")
print(f"Manual Answer Length: {len(answer_manual)} characters")