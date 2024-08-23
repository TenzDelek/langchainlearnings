from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# # we need to create the prompt template
print('prompt template ---')
template="tell a joke about {topic}"
prompttemplate=ChatPromptTemplate.from_template(template)

prompt=prompttemplate.invoke({"topic":"messi"}) #creates
result=model.invoke(prompt) #call for model

print(f'the prompt is: {prompt}')
print(result.content)

# creating multiples 
print('prompt multpile template ---')
template1="give {number} facts about {topic}"
multipletemp=ChatPromptTemplate.from_template(template1)
prompt1=multipletemp.invoke({"number":"two","topic":"tibetan"})
res1=model.invoke(prompt1)
print(res1.content)

print("message type------")
message=[("system","you are a comedian who tells facts about {topic}"), 
    ("human","tell me {number} facts"),]
prompttemp=ChatPromptTemplate.from_messages(message)
promt=prompttemp.invoke({"topic":"yak","number":"two"})
res=model.invoke(promt)
print(res.content)