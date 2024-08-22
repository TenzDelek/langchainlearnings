from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

res=model.invoke("what is the capital of tibet")

print("result: ")
print(res)
print(f'content only {res.content}')