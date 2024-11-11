import os
from dotenv import load_dotenv
from langchain_community.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

loader = TextLoader("./app/sample.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temparature":0, "max_length":512})
chain = load_qa_chain(llm, chain_type="stuff")

query = "What was said about the environment"
docs = db.similarity_search(query)
output = chain.run(input_documents=docs, question=query)

print(output)