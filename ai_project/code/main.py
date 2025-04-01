import streamlit as st
from dotenv import load_dotenv
import os

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain




def initilate():
    pdf_folder_path = r'Subjects\English\First Flight\papers'
    return pdf_folder_path

def loaders(pdf_folder_path):
    docs = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(pdf_folder_path, file))
            docs.extend(loader.load())  # Append all pages as separate documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(docs)
    return chunked_documents

def vector_store(docs):
    print("done 4")
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = Chroma(collection_name="example_collection", embedding_function=embeddings, persist_directory="./chroma_db")
    vector_store.add_documents(docs)
    print("done 5")
    
    query = str(input("What do you want to ask: "))
    retriever = vector_store.as_retriever()

    llm(retriever, query)



def llm(retriever, query):
    model = ChatOllama(model="llama3", base_url="http://localhost:11434")

    # Corrected prompt with 'context' variable
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a CBSE class X teacher and board checker. You have been given an English past year paper and are tasked to answer the user according to their question. Only use the information from the provided context."),
        ("user", "Context: {context}\n\nQuestion: {input}")
    ])

    # Creating the chain with proper context handling
    question_answer_chain = create_stuff_documents_chain(model, prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoke the chain with context from retrieved documents
    results = rag_chain.invoke({"input": query, "context": retriever.get_relevant_documents(query)})

    print(results["answer"])


    



    
    



def main():
    load_dotenv()
    print("done 1")
    paths = initilate()
    print("done 2")
    loader = loaders(paths)
    print("done 3")
    vector_store(loader)
    
    


main()