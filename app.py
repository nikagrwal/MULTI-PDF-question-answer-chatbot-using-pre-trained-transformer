import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

def getRawText(pdfs):
    raw_text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

def getTextChunks(raw_text):
    creating_text_chunks = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    text_chunks = creating_text_chunks.split_text(raw_text)
    return text_chunks

def creteVectorDatabase(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_database = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vector_database

def getConversationChain(vector_databse):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True)
    conversation_chain =ConversationalRetrievalChain.from_llm(llm =llm, 
                                      retriever = vector_databse.as_retriever(),
                                      memory = memory
                                    )
    return conversation_chain

def handleUserInput(user_input):
    response = st.session_state.conversation({'question' : user_input})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("user_input:", message.content)
        else:
            st.write("bot_output:", message.content)

def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Multi-pdf chat LLM")
    user_input = st.text_input("Input your question here:")
    if user_input:
        handleUserInput(user_input)

    with st.sidebar:
        st.subheader("Documemts")
        pdfs = st.file_uploader("Upload your pdfs here", accept_multiple_files= True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = getRawText(pdfs)

                text_chunks = getTextChunks(raw_text)

                vector_databse = creteVectorDatabase(text_chunks)

                st.session_state.conversation = getConversationChain(vector_databse)

if __name__ == '__main__':
    main()