from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

prompt=[(
    "system","you are a comedian who tells joke about {topic}"),
    ("human","tell me {jokecount} joke"),]
prompttemplate=ChatPromptTemplate.from_messages(prompt)
chain=prompttemplate | model | StrOutputParser()

res=chain.invoke({"topic":"messi","jokecount":"three"})
print(res)