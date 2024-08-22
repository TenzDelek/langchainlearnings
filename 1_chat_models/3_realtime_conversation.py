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
    humanmessage=HumanMessage(content=query)# we take the humanmessage as the input taken
    chathistory.append(humanmessage)#append it in the chathistory
    res=model.invoke(chathistory) #invoke all history
    response=res.content # the res by ai
    chathistory.append(AIMessage(content=response)) # assign the response to the chathistory
    print(f'AI: {response}')

print('--message-history--')
print(chathistory)
   
