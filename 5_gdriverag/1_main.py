import os
import pickle
from google_auth_oauthlib.flow import Flow
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_community import GoogleDriveLoader

# Load environment variables from .env
load_dotenv()

# Google Drive API setup
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
]

def authenticate_google_drive():
    creds = None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    secrets_path = os.path.join(current_dir, 'client_secrets.json')

    if os.path.exists(os.path.join(current_dir, 'token.json')):
        creds = Credentials.from_authorized_user_file(os.path.join(current_dir, 'token.json'), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(secrets_path, SCOPES)
            creds = flow.run_local_server(port=0)  # This opens a local server for authentication
        
        with open(os.path.join(current_dir, 'token.json'), 'w') as token:
            token.write(creds.to_json())
    
    return creds

# Authenticate and build the Drive service
creds = authenticate_google_drive()
drive_service = build('drive', 'v3', credentials=creds)

# Define the folder ID
folder_id = "1zhIAffPLuTaGX7XWkwlUzEtncI4ngek6"

# List files in the folder
results = drive_service.files().list(
    q=f"'{folder_id}' in parents",
    fields="files(id, name)"
).execute()

files = results.get('files', [])

# Print file list and ask for selection
print("Select a file:")
for i, file in enumerate(files):
    print(f"{i+1}. {file.get('name')}")

file_index = int(input("Enter the file number: ")) - 1
selected_file = files[file_index]

# Define the file ID
file_id = selected_file.get('id')
print(file_id)
# Load the file from Google Drive
loader = GoogleDriveLoader(
    service=drive_service,
    folder_id=folder_id,
    recursive=False,
    token_path=os.path.join(os.path.dirname(__file__), "token.json")
)
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Create or load the vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "langchaindemo")

if os.path.exists(persistent_directory):
    db = Chroma(persist_directory=persistent_directory, embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
else:
    db = Chroma.from_documents(splits, GoogleGenerativeAIEmbeddings(model="models/embedding-001"), persist_directory=persistent_directory)
    db.persist()

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Test the question answering chain
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))  # Use AIMessage instead of SystemMessage

# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()