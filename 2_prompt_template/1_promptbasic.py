from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


#creating normal prompttemplate
# template= "Tell me {times} joke about {topic}" 
# prompttemplate=ChatPromptTemplate.from_template(template)

# print('prompt template ---')
# promp=prompttemplate.invoke({"times":"two","topic":"messi"})
# print(promp)

#creating humanmessage and systemmessage template
message=[
    ("system","you are a comedian who tells joke about {topic}"), #when there is string manipulation we have to use tuples
   HumanMessage(content="tell me 3 joke"), #else we can use like before
]
ptemp=ChatPromptTemplate.from_messages(message)

print('prompt template ---')
promp=ptemp.invoke({"topic":"messi"})
print(promp)
