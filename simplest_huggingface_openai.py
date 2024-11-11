from dotenv import load_dotenv
from langchain import HuggingFaceHub
#from langchain_community.llms import OpenAI

load_dotenv()

#llm=OpenAI()
llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "min_length":512, "max_length":512})
prompt = "tell me a joke"
print(llm(prompt))