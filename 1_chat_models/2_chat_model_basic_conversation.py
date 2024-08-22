from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

mesages=[SystemMessage(content="solve this following math problem"),
    HumanMessage(content="what is 2+2?"),]

res=model.invoke(mesages)
print(f'the answer from ai is {res.content}')