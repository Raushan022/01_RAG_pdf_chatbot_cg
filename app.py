from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# create loader object
loader = PyPDFLoader("pdfs/starting_python.pdf")
# load documents 
documents = loader.load()
# print(len(documents))
# print(type(documents))
# print(documents[5])
# print(documents[5].page_content)

text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=1000,
   chunk_overlap=200
)
# create chunks
chunks = text_splitter.split_documents(documents)
# print(len(chunks))
# print(chunks[0].page_content)
# print(chunks[0].metadata)

# note: below line does not create embeddings yet. It only creates an object that knows:How to talk to OpenAI and How to generate embeddings
embeddings = OpenAIEmbeddings();

# generate first embedding
# vector = embeddings.embed_query(
#    "React is a JavaScript library"
# )
# print(type(vector))
# print(len(vector))
# print(vector[:10])

# create vector store
vectorstore = FAISS.from_documents(
   chunks,
   embeddings
)

print("Vector DB Created Successfully")

user_question = input(
    "Ask a question: "
)

results = vectorstore.similarity_search(
   user_question,
   k=3
)
# print(results[0].page_content)

# create model
llm = ChatOpenAI(
   model="gpt-4o-mini",
   temperature=0
)

context = "\n\n".join(
   [doc.page_content for doc in results]
)

prompt = f"""
Answer the question using ONLY the provided context.

context:
{context}

Question:
{user_question}

If the answer is not in the context, say
"I could not find the answer in the provided document."
 """
response = llm.invoke(prompt)
print(response.content)
