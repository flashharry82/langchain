import os
from dotenv import load_dotenv
from langchain_community.chains import LLMChain
from langchain_community.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv() # load en vars
documents = [] # array to put text into
loader1 = TextLoader("./app/cird81900.txt") # load first file
loader2 = TextLoader("./app/cird81910.txt") # load second file
documents.extend(loader1.load()) # add first file to documents array
documents.extend(loader2.load()) # add second file to documents array
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # set text splitter chunk size
docs = text_splitter.split_documents(documents) # split text in documents using text splitter
embeddings = HuggingFaceEmbeddings() # use huggingface embeddings
db = FAISS.from_documents(docs, embeddings) # convert text chunks to vectors and store in vector store

llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temparature":0, "max_length":512}) # select the model
chain = load_qa_chain(llm, chain_type="stuff") # create a model chain

query = "What was said about research" # set the query text
docs = db.similarity_search(query) # create a similarity search on the vector store contents using the query
output = chain.run(input_documents=docs, question=query) # run the chain and output to 'output' variable

print(output) # print the contents of the output variable