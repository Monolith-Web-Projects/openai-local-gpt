# main.py
import os
import warnings


from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

# disable instreuction stdout
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load documents
loader = TextLoader("data/data.txt", encoding="utf-8")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Create embeddings
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)

# Retrieval chain
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo"), retriever=retriever)

# Ask your question
while True:
    query = input("\nAsk something (or 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa.run(query)
    print("Answer:", result)
