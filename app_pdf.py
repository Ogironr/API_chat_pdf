# https://docs.streamlit.io/library/api-reference/chat/st.chat_message
#------------------------------------------

import streamlit as st
from openai import OpenAI

import os
import json
with open("config.json") as f:
    config = json.load(f)

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

#------------------------------------------

os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']

#------------------------------------------

def extract_data(feed):
    pdf1 = PdfReader(feed)

    raw_text = ''
    for i, page in enumerate(pdf1.pages):
        content = page.extract_text()
        if content:
            raw_text +=content

    text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 800,
            chunk_overlap  = 200,
            length_function = len,
        )

    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def make_request(question_input: str):
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    query = question_input #"¿Qué es la visión artificial?"
    docs = docsearch.similarity_search(query)
    response = chain.run(input_documents = docs, question = query) 
    return response

def clear_text():
    st.session_state["text"] = ""
    
#------------------------------------------

# Configuración de la aplicación Streamlit
st.title("Aplicación de Chat con PDF")
st.write("Sube un archivo PDF para consultar cualquier cosa.")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    docsearch = extract_data(uploaded_file)

response = False
prompt_tokens = 0
completion_tokes = 0
total_tokens_used = 0
cost_of_response = 0

if "question_input" not in st.session_state: 
    st.session_state.question_input = "" 
    st.session_state.widget = "" 

st.markdown("""---""")

st.session_state.question_input = st.session_state.widget 
st.session_state.widget = "" 
st.text_input("Enter text here", key="widget") 
question_input = st.session_state.question_input 

# question_input = st.text_input("Enter question")
rerun_button = st.button("Query")
st.markdown("""---""")

if question_input:
    response = make_request(question_input)
else:
    pass

if rerun_button:
    response = make_request(question_input)
else:
    pass

if "messages" not in st.session_state:   
    st.session_state.messages=[]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    
    if message["role"]=="user":
        with st.chat_message(message["role"], avatar="🧑"):
            st.markdown(message["content"])
    elif message["role"]=="assistant":
        with st.chat_message(message["role"], avatar="🤖"):
            st.markdown(message["content"])


# prompt=question_input
if question_input:
    # Display user message in chat message container
    st.chat_message("user", avatar="🧑").markdown(question_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question_input})

    # response2 = response.choices[0].message.content
    response2 = response
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response2)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response2})
    st.session_state["question_input"] = ""


#------------------------------------------
# streamlit run app_pdf.py  --theme.base "dark"
    
# ¿Qué es la visión artificial?
# cuando se hizo hincapié en la extracción de características?