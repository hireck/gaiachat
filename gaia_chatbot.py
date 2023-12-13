from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import openai
import hmac

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("ߘ堕ser not known or password incorrect")
    return False


if not check_password():
    st.stop()

key = st.secrets["OPENAIAPIKEY"]
headers = {
    "authorization":key,
    "content-type":"application/json"
    }
openai.api_key = key

st.title('Gaia chatbot')
question = st.text_input("Write a question about Gaia: ", key="input")

@st.cache_resource
def load_vectors():
    embedding_model = HuggingFaceEmbeddings()
    return FAISS.load_local("faiss_index", embedding_model)

vectorstore = load_vectors()

@st.cache_resource
def load_llm():
    return ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
llm = load_llm()
#question = 'Where is the GAIA spacecraft?'

#docs = vectorstore.similarity_search(question,k=5)

template = """Use the following pieces of context to answer the question at the end. Be helpful. Volunteer additional information where relevant, but keep it concise. Don't try to make up answers that are not supported by the context. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

if question:
    result = qa_chain({"query": question})
    st.write(result["result"])
    st.write('\n')
    st.write('Sources:')
    for num, rd in enumerate(result["source_documents"]):
        st.write(str(num+1)+') '+str(rd.metadata["title"]))
        section_info = []
        for item in rd.metadata:
            if item.startswith('Header'):
                section_info.append(rd.metadata[item])
        if rd.metadata.get("paragraph"):
            section_info.append('paragraph: '+rd.metadata["paragraph"])
        st.write('   (Section: '+', '.join(section_info)+')')
        st.write(rd.metadata["link"])
        #st.write(rd)
        st.write('\n')
    
    
# Check the result of the query

# Check the source document from where we 
# for rd in result["source_documents"]:
#     print(rd)
# print('\n')
# print(result["result"])