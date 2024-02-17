#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.chains import RetrievalQA
#from langchain.prompts import PromptTemplate
#from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import openai
import hmac
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
#from langchain.chains import LLMChain
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

#scores = model.predict([["My first", "sentence pair"], ["Second text", "pair"]])

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
        st.error("user not known or password incorrect")
    return False


if not check_password():
    st.stop()

apikey = st.secrets["OPENAIAPIKEY"]
headers = {
    "authorization":apikey,
    "content-type":"application/json"
    }
openai.api_key = apikey

st.title('Gaia chatbot')

# @st.cache_resource
# def load_vectors_latex():
#     embedding_model = HuggingFaceEmbeddings()
#     return FAISS.load_local("faiss_index", embedding_model)

# @st.cache_resource
# def load_vectors_html():
#     embedding_model = HuggingFaceEmbeddings()
#     return FAISS.load_local("faiss_index_html", embedding_model)

#vectorstore_latex = load_vectors_latex()
#vectorstore_html = load_vectors_html()

@st.cache_resource
def load_vectors():
    embedding_model = HuggingFaceEmbeddings()
    return FAISS.load_local("faiss_index_combined", embedding_model)

vectorstore = load_vectors()

@st.cache_resource
def load_llm():
    return ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
llm = load_llm()
#question = 'Where is the GAIA spacecraft?'

#docs = vectorstore.similarity_search(question,k=5)

msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

template = """You are a Gaia expert at the ESA helpdesk. You task is to help researchers learn about Gaia and use Gaia data products effectively in their work. 
Use the following pieces of retrieved information to answer the user's question. Be helpful. Volunteer additional information where relevant, but keep it concise. Don't try to make up answers that are not supported by the context. 

Retrieved information:
{context}

Preceeding conversation:
{conversation}

Question: {question}
Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
#     memory=memory
# )

#prompt = PromptTemplate(input_variables=["context", "question"], template=template)
#llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

#prompt = ChatPromptTemplate.from_template(template)
#chain = prompt | llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#    # | StrOutputParser()
# )#


for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

#question = st.text_input("Write a question about Gaia: ", key="input")
    
    

def add_sources(docs):
    lines = []
    #lines.append('\nSources:')
    for num, rd in enumerate(docs): #result["source_documents"]):
        doc_info = []
        doc_info.append(str(num+1)+') '+str(rd.metadata["title"]))
        section_info = []
        for item in rd.metadata:
            if item.startswith('Header'):
                section_info.append(rd.metadata[item])
        if rd.metadata.get("paragraph"):
            section_info.append('paragraph: '+rd.metadata["paragraph"])
        doc_info.append('   (Section: '+', '.join(section_info)+')')
        doc_info.append('\n'+rd.metadata["link"])
        lines.append(''.join(doc_info))
    text = '\"\"\"'+'\n'.join(lines)+'\"\"\"'
    return '\n'.join(lines)

if user_input := st.chat_input():
    st.chat_message("human").write(user_input)
    #retrieved = vectorstore_latex.similarity_search(user_input,k=15)
    #retrieved_html = vectorstore_html.similarity_search(user_input,k=15)
    #retrieved.extend(retrieved_html)
    retrieved = vectorstore.similarity_search(user_input,k=25)
    cross_inp = [[user_input, d.page_content] for d in retrieved]
    cross_scores = cross_encoder.predict(cross_inp)
    scored = [(score, d) for score, d in zip(cross_scores, retrieved) if score > 0]
    if scored:
        reranked = sorted(scored, key=lambda tup: tup[0], reverse=True)
        docs = [r[1] for r in reranked[:7]]
    else:
        reranked = sorted(scored, key=lambda tup: tup[0]) #if there are no positive score, take the 2 least negative ones
        docs = [r[1] for r in reranked[:2]]
   #context = format_docs(docs)
    #prompt_value = prompt.invoke({"context":context, "question":question})
    prev_conv = '\n'.join([msg.type+': '+msg.content for msg in msgs.messages[-2:]])
    full_prompt = template.format(context=format_docs(docs), question=user_input, conversation=prev_conv)
    print(full_prompt)
    result = llm.invoke(full_prompt)
    with st.chat_message("ai"):
        st.write(result.content)#+add_sources(docs))
        expander = st.expander("See sources")
        expander.write(add_sources(docs))
    msgs.add_user_message(user_input)
    msgs.add_ai_message(result.content)
   
    
    
# Check the result of the query

# Check the source document from where we 
# for rd in result["source_documents"]:
#     print(rd)
# print('\n')
# print(result["result"])