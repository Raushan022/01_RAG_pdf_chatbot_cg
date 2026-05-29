import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# Create embedding model
embeddings = OpenAIEmbeddings()

# FAISS folder name
FAISS_PATH = "faiss_db"

# Check if FAISS DB already exists
if os.path.exists(FAISS_PATH):

    print("Loading existing FAISS database...")

    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    print("Creating new FAISS database...")

    # Load PDF
    loader = PyPDFLoader("pdfs/starting_python.pdf")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save FAISS locally
    vectorstore.save_local(FAISS_PATH)

    print("Vector DB Created and Saved Successfully!")

# Ask question
user_question = input("Ask a question: ")

# Similarity search
results = vectorstore.similarity_search(
    user_question,
    k=3
)

# Create LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Create context
context = "\n\n".join(
    [doc.page_content for doc in results]
)

# Prompt
prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{user_question}

If the answer is not in the context, say:
"I could not find the answer in the provided document."
"""

# Get response
response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)