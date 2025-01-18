import os
import streamlit as st
from dotenv import load_dotenv
import gdown
import zipfile

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

# Retrieve API key from environment
groq_api_key = os.getenv('GROQ_API_KEY')

# Constants for Google Drive
GDRIVE_FILE_ID = '1BTF3EuWKHf6pIhOdZej64ON4lm0zHaAo'  # Replace with your actual FILE_ID
GDRIVE_DOWNLOAD_URL = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
ZIP_PATH = 'faiss_health.zip'
EXTRACT_DIR = 'faiss health'


###############################################################################
# 1. Initialize Session State
###############################################################################
def initialize_components():
    
    # Function to download and extract FAISS embeddings
    @st.cache_resource
    def download_and_extract_embeddings():
        if not os.path.exists(EXTRACT_DIR):
            try:
                # Download the zip file using gdown
                gdown.download(url=GDRIVE_DOWNLOAD_URL, output=ZIP_PATH, quiet=False)
            except Exception as e:
                st.error(f"❌ Error downloading embeddings: {e}")
                return False

            try:
                # Extract the zip file
                with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall('.')  # Extract to current directory
                
                # Remove the zip file after extraction to save space
                os.remove(ZIP_PATH)
            except Exception as e:
                st.error(f"❌ Error extracting embeddings: {e}")
                return False

        # Debugging: List contents of EXTRACT_DIR to verify extraction
        if os.path.exists(EXTRACT_DIR):
            # Temporary debugging statements
            st.write("✅ FAISS embeddings are present.")
            st.write("Contents of 'faiss health':")
            st.write(os.listdir(EXTRACT_DIR))
        else:
            st.write("❌ FAISS embeddings directory does not exist after extraction.")
        
        return True

    # Initialize embeddings
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    # Download and extract embeddings if not already done
    embeddings_ready = download_and_extract_embeddings()

    if embeddings_ready:
        # Initialize FAISS vector store
        if "vectors" not in st.session_state:
            try:
                st.session_state.vectors = FAISS.load_local(
                    EXTRACT_DIR,
                    st.session_state.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                st.error(f"❌ Error loading FAISS vector store: {e}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Rest of your initialization code...

    if "history_aware_chain" not in st.session_state and embeddings_ready:
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7
        )

        # Define system prompt
        system_prompt = """
        Answer the following question based **only** on the provided context. 
        Think step-by-step carefully before providing a detailed answer.
        Explain the answers in brief and the logic behind it as well, 
        but do not make up any information on your own.
        <context>
        {context}
        </context>
        """

        # Create ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Question: {input}")
        ])

        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Initialize retriever
        retriever = st.session_state.vectors.as_retriever()

        # Define retriever prompt
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look "
                     "up in order to get information relevant to the conversation")
        ])

        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=retriever_prompt
        )

        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # Store in session state
        st.session_state.history_aware_chain = retrieval_chain

###############################################################################
# 2. Streamlit Layout (Header, Sidebar, Footer)
###############################################################################
def header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Health-related logo from Flaticon (Stethoscope Icon)
        st.image(
            "https://cdn-icons-png.flaticon.com/512/833/833472.png",
            width=80,
            # caption="Health Logo"
        )
        st.markdown(
            "<h2 style='text-align: center; color: #2E86C1;'>Health Query Assistant</h2>",
            unsafe_allow_html=True
        )
    with col3:
        # Fitness-related GIF from Giphy (Running GIF)
        st.image(
            "https://media.giphy.com/media/3o7TKKlj7BTQGna32o/giphy.gif?cid=790b7611m7tl6f2u533921p93ywg6q08er5ntzzi63f9em1w&ep=v1_gifs_search&rid=giphy.gif&ct=g",
            width=80,
            # caption="Fitness GIF"
        )


def sidebar():
    st.sidebar.title("📋 About")
    st.sidebar.info(
        """
        **Health Query Assistant** is a Retrieval-Augmented Generation (RAG) 
        system designed to provide detailed answers to your health-related 
        questions based on the provided documents.

        **Features:**
        - Answer health-related queries.
        - Provides document similarity search.
        - Utilizes advanced language models for accurate responses.
        """
    )
    # st.sidebar.image("https://media.giphy.com/media/26BRuo6sLetdllPAQ/giphy.gif", width=200)


def footer():
    st.markdown("""
        <hr>
        <div style='text-align: center; color: #95A5A6; font-size: 12px;'>
            &copy; 2025 Health Query Assistant. All rights reserved.
        </div>
        """, unsafe_allow_html=True)


###############################################################################
# 3. Display Conversation
###############################################################################
def show_conversation():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])


###############################################################################
# 4. Handle New Query
###############################################################################
def handle_query(query):
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Prepare chat history in required format
    formatted_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in st.session_state.chat_history
    ]

    # Invoke the retrieval chain
    response = st.session_state.history_aware_chain.invoke({
        "chat_history": formatted_history,
        "input": query,
    })

    # Extract the answer
    answer = response.get("answer", "I'm sorry, I couldn't find an answer to that.")

    # Append assistant's answer to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    return answer


###############################################################################
# 5. Main Streamlit App
###############################################################################
def main():
    st.set_page_config(
        page_title="Health Query Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize components
    initialize_components()

    # Render header and sidebar
    header()
    sidebar()

    # Apply CSS to make the conversation area scrollable and prevent centering
    st.markdown(
        """
        <style>
        /* Make the main area take full height */
        .stApp {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        /* Conversation area styling */
        .conversation {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #ffffff;
            margin-bottom: 20px;
        }
        /* Input box fixed at bottom */
        .input-box {
            position: fixed;
            bottom: 0;
            width: 100%;
            padding: 10px;
            background-color: #f0f2f6;
            border-top: 1px solid #ddd;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 🩺 Ask Your Health-Related Questions")

    # Conversation container
    st.markdown('<div class="conversation">', unsafe_allow_html=True)
    show_conversation()
    st.markdown('</div>', unsafe_allow_html=True)

    # Input at the bottom using st.chat_input
    user_input = st.chat_input(
        placeholder="e.g., How many calories are burned during running?"
    )

    if user_input:
        with st.spinner("🔍 Fetching answer..."):
            answer = handle_query(user_input)

        # Display assistant's answer
        with st.chat_message("assistant"):
            st.markdown(f"### **Answer:** 📝\n{answer}")

    footer()


if __name__ == "__main__":
    main()
