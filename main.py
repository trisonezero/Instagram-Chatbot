import boto3
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockLLM, BedrockEmbeddings, ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage

#Configure LLM
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

#Prompt template
template = """
You are a helpful AI assistant. Use the following pieces of context to answer the input question at the end.

{context}

{input}

Assistant:
"""

#Configure prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "input"],
    template=template
)

# Create runnable sequence
question_chain = prompt_template | llm

# Get question, retrieve context and generate answer
while True:
    question = input("Enter your question: ")
    context = vector_search(question)
    messages = [
        SystemMessage(content="You are a helpful AI assistant. Use the following pieces of context to answer the input question at the end.\n\n" + context),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    print("Assistant:", response.content)
    print("\n")
