import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization =True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt= PromptTemplate(template=custom_prompt_template,input_variable=["context","question"])
    return prompt

def load_llm(huggingface_repo_id,HUGGINGFACE_API_KEY):
    llm=HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        temperature =0.5,
        model_kwargs={
            "token":HUGGINGFACE_API_KEY,
            "max_length": "512"
                    })
    
    return llm

def main():
    st.title("Ask Microbio Bot!")
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt= st.chat_input("Pass Your Prompt Here")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})

        CUSTOM_PROMPT_TEMPLATE= """
            use the pieces of information provided in the context to answer uwer's question.
            if you don't know the answer, just say that you dont know, don't try to make up an answer.
            Don't provide anythin out of the given context
            context:{context}
            Question:{question}
            """
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HUGGINGFACE_API_KEY=os.environ.get("HUGGINGFACE_API_KEY")
    
        #llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID,HUGGINGFACE_API_KEY=HUGGINGFACE_API_KEY)
        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID,HUGGINGFACE_API_KEY=HUGGINGFACE_API_KEY),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt":set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response =qa_chain.invoke({'query':prompt})
            result=response["result"]
            # source_documents=response["source_documents"]
            # result_to_show=result+"\n\nSource Docs:\n"+str(source_documents)

            # response= "Hi, I am Medibot!"
            st.chat_message("assistant").markdown(result)
        
            st.session_state.messages.append({'role':'assistant','content':response})
        except Exception as e:
            st.error(f"Error:{str(e)}")




if __name__ == "__main__":
    main()

