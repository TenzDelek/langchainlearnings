import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage
load_dotenv()
# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "langchaindemo")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#load existing vector store
db = Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

query="how can i learn more langchain"

#retrive based on query
retriever=db.as_retriever(search_type="similarity",search_kwargs={"k":1})
retriever_docs=retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(retriever_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input=(
    "here are some documents that might help answer the question: "
    +query
    +"\n\n".join([doc.page_content for doc in retriever_docs])
    +"\n\nPlease provide an answer based on the provided documents If the answer is not found in the documents, respond with 'I'm not sure'. "
)

model=ChatGoogleGenerativeAI(model="gemini-1.5-pro")

#define the message
message=[
    SystemMessage(content="you are a helpful assistant"),
    HumanMessage(content=combined_input),
]

#invoke the model with the input
res=model.invoke(message)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(res)
print("Content only:")
print(res.content)