import getpass
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=google_api_key,
    # other params...
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)


st.title('Language translator')
# liste des langues
languages = ["Français", "Anglais", "Allemand", "Espagnol"]

selected_language_i = st.selectbox("Choisissez la langue à traduire:", languages)
selected_language_o = st.selectbox("Choisissez la langue de traduction:", languages)

input_text=st.text_input("Write the sentence in {selected_language_i} and it will be translated in {selected_language_o}")



# chain = prompt | llm
output_parser=StrOutputParser()

chain=prompt|llm|output_parser  

if input_text:
    st.write(chain.invoke(
    {
        "input_language": selected_language_i,
        "output_language": selected_language_o,
        "input": input_text, 
    }
))

      