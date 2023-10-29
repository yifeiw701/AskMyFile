import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os
import pinecone
from docx import Document
import tempfile

def display_csv_chat():
    load_dotenv()

    st.write(css, unsafe_allow_html=True)

    if "csv_agent" not in st.session_state:
        st.session_state.csv_agent = None
    if "csv_chat_history" not in st.session_state:
        st.session_state.csv_chat_history = []

    st.header("Chat with your CSV  :chart:")
    user_question = st.text_input("Ask a question:) ")
    if user_question and st.session_state.csv_agent:
        handle_csv_userinput(user_question)

    with st.sidebar:
        st.subheader("Your CSV")
        csv_file = st.file_uploader("Upload a CSV file", type="csv")
        if csv_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temporary_file:
                temporary_file.write(csv_file.getvalue())

            # initialize agent
            st.session_state.csv_agent = create_csv_agent(OpenAI(temperature=0), path=temporary_file.name, verbose=True)
            # delete temp file
            os.unlink(temporary_file.name)


def display_doc_chat():
    load_dotenv()

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your documents :four_leaf_clover:")
    user_question = st.text_input("Ask a question:) ")
    if user_question:
        handle_doc_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload PDFs or DOCX files here and click on 'Process'", 
                                          type=["pdf", "docx"], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = ""
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/pdf":
                        raw_text += get_pdf_text([uploaded_file])
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        raw_text += get_doc_text([uploaded_file])

                text_chunks = get_text_chunks(raw_text)
                PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
                PINECONE_ENV = os.getenv('PINECONE_ENV')
                pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


def get_doc_text(doc_docs):
    text = ""
    for doc in doc_docs:
        doc_reader = Document(doc)
        for para in doc_reader.paragraphs:
            text += para.text
    return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_texts(texts=text_chunks, embedding=embeddings, index_name='docs-chat')
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_doc_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i in range(len(st.session_state.chat_history)-2, -2, -2):
        st.write(user_template.replace(
            "{{MSG}}", st.session_state.chat_history[i].content), unsafe_allow_html=True) 
        st.write(bot_template.replace(
            "{{MSG}}", st.session_state.chat_history[i+1].content), unsafe_allow_html=True) 
        
def handle_csv_userinput(user_question):
    response = st.session_state.csv_agent.run(user_question)
    st.session_state.csv_chat_history.insert(0, {'user': user_question, 'bot': response})

    for chat in st.session_state.csv_chat_history:
        st.write(user_template.replace("{{MSG}}", chat['user']), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", chat['bot']), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Document Chatbot", page_icon=":four_leaf_clover:")

    choice = st.sidebar.selectbox("", ["Chat with PDF/DOCS", "Chat with CSV"])

    if choice == "Chat with CSV":
        display_csv_chat()
    elif choice == "Chat with PDF/DOCS":
        display_doc_chat()

if __name__ == '__main__':
    main()
