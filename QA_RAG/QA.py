import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


st.title("RAG Application built on Gemini Model")

UPLOAD_DIR = "uploaded_pdfs"

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    
    # Enregistrer le fichier PDF dans le répertoire spécifié
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF and split into chunks
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Create FAISS vector store
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.from_documents(documents=docs, embedding=embedding)

    # Set up the retriever
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Set up the language model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

    # Define the prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Streamlit UI
    query = st.chat_input("Say something: ") 
    if query:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query})
        st.write(response["answer"])
else:
    st.write("Please upload a PDF file.")
