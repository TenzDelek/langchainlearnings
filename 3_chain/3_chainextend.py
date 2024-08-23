from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

prompt=[("system","you are a comedian who tells joke about {topic}"),
        ("human","tell me {jokecount} joke")]

prompttemplate=ChatPromptTemplate.from_messages(prompt)
# with runnable lambda we can also use api calls
uppercase=RunnableLambda(lambda x: x.upper()) #extended task
countword=RunnableLambda(lambda x:f'word count {len(x.split())}\n{x}') # check length then at last after tue \n we print it

chain=prompttemplate | model | StrOutputParser() | uppercase | countword
res=chain.invoke({"topic":"lawyer","jokecount":2})
print(res)