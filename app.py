import os
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# 1. Cross-Platform Path Configuration
# ==============================================================================
# Use pathlib.Path so directory resolution works seamlessly across Linux/Ubuntu,
# WSL, macOS, and full Windows without hardcoded OS-specific slashes.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIRECTORY = BASE_DIR / "data"
CHROMA_DIRECTORY = BASE_DIR / ".chroma_db"

# ==============================================================================
# 2. Streamlit Page Setup & Header
# ==============================================================================
st.set_page_config(
    page_title="Taiwan International Student Healthcare & Admin Explainer",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Taiwan International Student Healthcare & Admin Explainer")
st.markdown(
    "Ask questions about **National Health Insurance (NHI)**, **ARC**, **health checkups**, "
    "and **administrative procedures** in Taiwan—grounded strictly in official PDF documents."
)


# ==============================================================================
# 3. API Key Resolution (Cloud Secrets -> Env Var -> Sidebar Input)
# ==============================================================================
def get_api_key() -> str:
    """Resolve the Google Gemini API key across local and cloud environments."""
    # Priority 1: Streamlit secrets (.streamlit/secrets.toml or Streamlit Cloud)
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    # Priority 2: System environment variable
    env_key = os.environ.get("GOOGLE_API_KEY", "")
    if env_key:
        return env_key

    # Priority 3: Manual entry via sidebar input for quick local testing
    with st.sidebar:
        st.header("🔑 Configuration")
        return st.text_input(
            "Enter your Google Gemini API Key",
            type="password",
            help="Stored in session only. For deployment, set GOOGLE_API_KEY in Streamlit Cloud Secrets.",
        )


api_key = get_api_key()
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key


# ==============================================================================
# 4. Helper Functions for Context Formatting
# ==============================================================================
def format_docs(docs) -> str:
    """Format retrieved PDF chunks with source filename and 1-indexed page numbers."""
    formatted = []
    for doc in docs:
        source_name = Path(doc.metadata.get("source", "Unknown")).name
        page_num = doc.metadata.get("page", 0) + 1
        formatted.append(f"\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# ==============================================================================
# 5. Vector Database & RAG Pipeline Initialization (Cached)
# ==============================================================================
@st.cache_resource(show_spinner="Indexing official PDF documents...")
def get_rag_chain(api_key_token: str):
    """Load PDFs from data/, chunk text, build Chroma vector store, and return RAG chain."""
    # Ensure the data folder exists and scan for official PDF documents
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(DATA_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        return None, []

    # Load all pages from each PDF file
    documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    # Split documents into overlapping chunks to preserve context across boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(documents)

    # Initialize Gemini embedding model and persist vectors locally in ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key_token,
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIRECTORY),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Initialize Gemini LLM with low temperature for factual, grounded responses
    llm = ChatGoogleGenerativeAI(
        model="gemini-4-hydrogen",
        temperature=0.1,
        google_api_key=api_key_token,
    )

    # Strict grounding prompt to prevent hallucination and enforce citations
    prompt = ChatPromptTemplate.from_template(
        """You are a reliable administrative and healthcare assistant for international students in Taiwan.
Answer the user's question using ONLY the provided official document context below.
If the answer cannot be found in the context, state clearly that the provided documents do not contain that information—do not guess or hallucinate.
Always cite the source filename and page number for your statements.

Context:
{context}

Question: {question}

Answer:"""
    )

    # Assemble LCEL (LangChain Expression Language) RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, [p.name for p in pdf_files]


# ==============================================================================
# 6. Sidebar Document Inventory
# ==============================================================================
with st.sidebar:
    st.subheader("📂 Loaded Official Documents")
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    available_pdfs = sorted(DATA_DIRECTORY.glob("*.pdf"))
    if available_pdfs:
        for pdf in available_pdfs:
            st.write(f"- `{pdf.name}`")
    else:
        st.warning(
            f"No PDFs found in `{DATA_DIRECTORY.name}/`. Add official PDFs and rerun."
        )


# ==============================================================================
# 7. Pre-flight Checks (API Key & Reference PDFs)
# ==============================================================================
if not api_key:
    st.info(
        "Please provide a `GOOGLE_API_KEY` via `.streamlit/secrets.toml`, environment variable, or the sidebar to start."
    )
    st.stop()

rag_chain, indexed_files = get_rag_chain(api_key)

if rag_chain is None:
    st.error(
        f"No PDF documents found in `{DATA_DIRECTORY}`. "
        "Please place official PDF files inside the `data/` folder and refresh the page."
    )
    st.stop()


# ==============================================================================
# 8. Chat Session State & History Rendering
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==============================================================================
# 9. User Chat Input & RAG Response Generation
# ==============================================================================
# User Chat Input: Capture user query from the Streamlit chat input bar
if user_query := st.chat_input(
    "Ask about Taiwan NHI, ARC, health exams, or student procedures..."
):
    # Append and display user message in chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate RAG Response: Query the vector store, call Gemini, and render output
    with st.chat_message("assistant"):
        with st.spinner("Searching official documents..."):
            response = rag_chain.invoke(user_query)
            st.markdown(response)

    # Save assistant response to session state for conversation persistence
    st.session_state.messages.append({"role": "assistant", "content": response})
