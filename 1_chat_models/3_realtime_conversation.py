from dotenv import load_dotenv
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
chathistory=[]

systemmessage=SystemMessage(content="you are a AI assistant")
chathistory.append(systemmessage)

while True:
    query=input("Tenzin(User): ")
    if query.lower()=='exit':
        break
    humanmessage=HumanMessage(content=query)
    chathistory.append(humanmessage)
    res=model.invoke(chathistory)
    response=res.content
    chathistory.append(AIMessage(content=response))
    print(f'AI: {response}')

print('--message-history--')
print(chathistory)
   
