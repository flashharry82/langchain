from dotenv import load_dotenv
#from langchain_community.chat_models import ChatOpenAI
from langchain import HuggingFaceHub
from langchain_community.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.chains import LLMChain
from langchain_community.schema import BaseOutputParser

load_dotenv()

class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")

start_template = "Create a list of 5 unique items using the following category."
system_message_prompt = SystemMessagePromptTemplate.from_template(start_template)
input_template1 = "{text}"
message_prompt1 = HumanMessagePromptTemplate.from_template(input_template1)

#llm=ChatOpenAI()
llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temparature":0, "max_length":512})

input_template2 = "Return an original meal idea using all the ingredients."
message_prompt2 = HumanMessagePromptTemplate.from_template(input_template2)
prompt = ChatPromptTemplate.from_messages([system_message_prompt, message_prompt1, message_prompt2])

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run('healthy foods'))