from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.schema import Document
import json
from langchain_community.vectorstores.utils import filter_complex_metadata


embeddings = OllamaEmbeddings(
    model="llama3",
)



loader = JSONLoader(
    file_path="ai_project\Subjects\English\First Flight\papers\sample.json",
    jq_schema=".[].question",
    text_content=False,
)

docs = loader.load()

loader = JSONLoader(
    file_path="ai_project\Subjects\English\First Flight\papers\sample.json",
    jq_schema="del(.[].question)",
    text_content=False,
)

metadata = loader.load()




for i in metadata:
    data_list = json.loads(i.page_content)


increment = 0 
for i in docs:
    i.metadata = data_list[increment]
    increment += 1

db = Chroma.from_documents(filter_complex_metadata(docs), embeddings)

answer = db.similarity_search("show me some Questions on Lencho")
for i in answer:
    print (i.page_content)
