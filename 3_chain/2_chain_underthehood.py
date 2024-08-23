from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-pro")
prompt=[(
    "system","you are a poet who tells poem about {topic}"),
    ("human","tell me {poemline} line peom")]
promptemplate=ChatPromptTemplate.from_messages(prompt)

# chain at high level other option is the one that we did at 1_basic with the use of |
# understand  runable as a task
formatprompt=RunnableLambda(lambda x:promptemplate.format_prompt(**x)) # ** unpack the dictionary
invokemodel=RunnableLambda(lambda x:model.invoke(x.to_messages()))
praseoutput=RunnableLambda(lambda x:x.content)

chain=RunnableSequence(first=formatprompt,middle=[invokemodel],last=praseoutput)
#first and last is in normal rest in the middle is past as a list
#if you have two then use first and Last
response=chain.invoke({"topic":"himalaya","poemline":"three"})
print(response)