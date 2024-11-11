from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI

load_dotenv()
llm = OpenAI()
chat_model = ChatOpenAI()

print(chat_model.predict("hi!"))