import os

import streamlit as st
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Configuration
st.set_page_config(
    page_title="Taiwan Healthcare & Admin Explainer",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Taiwan Healthcare & Admin Document Explainer")
st.markdown(
    "An AI assistant powered by **RAG (Retrieval-Augmented Generation)** and"
    " **Google Gemini** to help international students navigate Taiwan's"
    " healthcare and university policies."
)

# Constants
PERSIST_DIRECTORY = "./chroma_db"
DATA_DIRECTORY = "./data"

# 2. Sidebar Configuration for API Key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_Key_input = st.text_input("Enter Google Gemini API Key", type="password")

    if api_Key_input:
        os.environ["GOOGLE_API_KEY"] = api_Key_input

    st.markdown("---")
    st.markdown("### 📂 Document Status")
    if os.path.exists(DATA_DIRECTORY):
        pdf_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(".pdf")]
        st.write(f"Found **{len(pdf_files)}** PDF(s) in `data/` folder.")
        for pdf in pdf_files:
            st.text(f"- {pdf}")
    else:
        st.warning("`data/` folder missing. Please create it and add PDFs.")

    st.markdown("---")
    st.markdown(
        "### About\nThis app restricts Gemini to answer **strictly** from your"
        " curated PDF documents, preventing hallucinations and ensuring local"
        " policy accuracy."
    )


# 3. Initialize Vector Database & RAG Pipeline (Cached for performance)
@st.cache_resource
def get_rag_chain():
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
        return None

    pdf_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(".pdf")]
    if not pdf_files:
        return None

    # Load and split documents
    loader = PyPDFDirectoryLoader(DATA_DIRECTORY)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embeddings and Vector Store (ChromaDB)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=PERSIST_DIRECTORY
    )

    # Retriever (fetches top 3 relevant chunks)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # LLM Engine (Gemini 1.5 Flash)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    # Strict System Prompt Guardrail
    system_prompt = (
        "You are an official administrative and healthcare assistant for"
        " international students in Taiwan.\nUse the provided context chunks"
        " below to answer the user's question.\nIf the answer cannot be found"
        " within the provided context, you must explicitly state: 'I cannot find"
        " this information in the official documents provided.' Do not use"
        " outside knowledge.\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


# 4. Main Application Logic
if "GOOGLE_API_KEY" in os.environ:
    try:
        with st.spinner("Initializing vector database and processing documents..."):
            rag_chain = get_rag_chain()

        if rag_chain is None:
            st.error(
                "⚠️ No PDF documents found in the `data/` folder. Please add at least"
                " one official PDF guide (e.g., NHI foreigner guide) to use the app."
            )
        else:
            # Initialize Chat History
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display Prior Messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # User Chat Input
            if user_query := st.chat_input(
                "Ask a question about Taiwan healthcare, NHI, or admin procedures..."
            ):
                st.session_state.messages.append(
                    {"role": "user", "content": user_query}
                )
                with st.chat_message("user"):
                    st.markdown(user_query)

                # Generate RAG Response
                with st.chat_message("assistant"):
                    with st.spinner(
                        "Searching official documents and generating response..."
                    ):
                        response = rag_chain.invoke({"input": user_query})
                        answer = response["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )

    except Exception as e:
        st.error(f"An error occurred while setting up the RAG pipeline: {str(e)}")
else:
    st.info(
        "👈 Please enter your Google Gemini API Key in the sidebar to initialize"
        " the application."
    )
