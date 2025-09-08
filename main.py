import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

#Configure streamlit app
st.set_page_config(page_title="Dark Souls 1 Dialouge Bot", page_icon=":⚔️:")
st.title("Dark Souls 1 Dialouge Bot :⚔️:")


#Configure LLM
@st.cache_resource
def configure_llm():
    client = boto3.client(
        'bedrock-runtime',
    )
    
    model_id = "global.anthropic.claude-sonnet-4-20250514-v1:0"
    llm = ChatBedrockConverse(
        client=client,
        model_id=model_id,
        temperature=0.7,
        max_tokens=1024
    )
    return llm

#Configure Vectorstore
@st.cache_resource
def configure_vectorstore(filename):
    global vectorstore_faiss
    client = boto3.client(
        'bedrock-runtime',
    )
    
    embeddings = BedrockEmbeddings(
        client=client
    )
    
    # Load documents from PDF
    loader = PyPDFLoader(filename)
    documents = loader.load_and_split()
    
    # Create FAISS vector store
    vectorstore_faiss = FAISS.from_documents(documents, embeddings)

    return vectorstore_faiss

#Vector Search
def vector_search(query):
    docs = vectorstore_faiss.similarity_search(query)
    
    info = ''.join([doc.page_content for doc in docs])

    return info

#Configure llm and vectorstore
llm = configure_llm()
vectorstore_faiss = configure_vectorstore("darksouls1_dialouge.pdf")

#Setup chat history
message_history = StreamlitChatMessageHistory(key = "chat_history")
if len (message_history.messages) == 0:
    message_history.add_ai_message("Hello! I am your Dark Souls 1 Dialouge Bot. Ask questions about Dark Souls 1 Dialouge")

#Show chat messages
for msg in message_history.messages:
    st.chat_message(msg.type).write(msg.content)

#Get user input and gererate response
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    #Similarity search and generate response
    context = vector_search(prompt)
    messages = [
        SystemMessage(content="You are a helpful AI chatbot desinged to answer questions about the dialouge in the video game Dark Souls 1. Use the following pieces of context to answer the input question at the end.\n\n" + context),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    message_history.add_user_message(prompt)
    message_history.add_ai_message(response.content)

    #Display AI response
    st.chat_message("assistant").write(response.content)