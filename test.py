from langchain_community.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

llm=OpenAI()
prompt = "tell me a joke"
print(llm(prompt))