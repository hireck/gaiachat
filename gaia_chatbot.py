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
from langchain_core.messages.base import BaseMessage
#from sentence_transformers import SentenceTransformer
import re

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

#scores = model.predict([["My first", "sentence pair"], ["Second text", "pair"]])

# def check_password():
#     """Returns `True` if the user had a correct password."""

#     def login_form():
#         """Form with widgets to collect user information"""
#         with st.form("Credentials"):
#             st.text_input("Username", key="username")
#             st.text_input("Password", type="password", key="password")
#             st.form_submit_button("Log in", on_click=password_entered)

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if st.session_state["username"] in st.secrets[
#             "passwords"
#         ] and hmac.compare_digest(
#             st.session_state["password"],
#             st.secrets.passwords[st.session_state["username"]],
#         ):
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # Don't store the username or password.
#             del st.session_state["username"]
#         else:
#             st.session_state["password_correct"] = False

#     # Return True if the username + password is validated.
#     if st.session_state.get("password_correct", False):
#         return True

#     # Show inputs for username + password.
#     login_form()
#     if "password_correct" in st.session_state:
#         st.error("user not known or password incorrect")
#     return False


# if not check_password():
#     st.stop()

apikey = st.secrets["OPENAIAPIKEY"]
headers = {
    "authorization":apikey,
    "content-type":"application/json"
    }
openai.api_key = apikey



st.title('Gaia chatbot')


@st.cache_resource
def load_vectors():
    embedding_model =  HuggingFaceEmbeddings()#model_name="sentence-transformers/all-MiniLM-L12-v2")#, encode_kwargs={"normalize_embeddings": True},)
    #embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large", encode_kwargs={"normalize_embeddings": True},)#"msmarco-bert-base-dot-v5")
    #embedding_model =  HuggingFaceEmbeddings(model_name="thenlper/gte-large", encode_kwargs={"normalize_embeddings": True}SentenceTransformer('all-MiniLM-L6-v2')
    #embedmodel.max_seq_length = 512
    return FAISS.load_local("faiss_index_combined", embedding_model, allow_dangerous_deserialization=True)

vectorstore = load_vectors()

@st.cache_resource
def load_gpt3_5():
    #return ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
    return ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

@st.cache_resource
def load_gpt4():
    return ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    
gpt3_5 = load_gpt3_5()
gpt4 = load_gpt4()
#question = 'Where is the GAIA spacecraft?'

#docs = vectorstore.similarity_search(question,k=5)

msgs = StreamlitChatMessageHistory(key="langchain_messages")
#memory = ConversationBufferMemory(chat_memory=msgs)
#message_history = []


if len(msgs.messages) == 0:
    new_msg = BaseMessage(type='ai', content="How can I help you?")
    msgs.add_message(new_msg)
    #msgs.add_ai_message("How can I help you?")

template = """You are a Gaia expert at the ESA helpdesk. Your task is to help researchers learn about Gaia and use Gaia data products effectively in their work. 
Use the following pieces of retrieved information to answer the user's question. Be helpful. Volunteer additional information where relevant, but keep it concise. Don't try to make up answers that are not supported by the retrieved information. If the retrieved documents do not contain sufficient information to answer the question, say so.
Avoid using latex commands in your answer, use the intended symbols instead.  

Include references in your answer to the documents you used, to indicate where the information comes from. The documents are numbered. Use those numbers to refer to them. Use the term 'Document' followed by the number, e.g. '(Document 1)' or '(Document 2, Document 5)' when citing multiple documents. Do not list the sources below your answer. They will be provided by a different component.

Retrieved information:
{context}

Preceeding conversation:
{conversation}

Question: {question}
Helpful Answer:"""

contextualizing_template = """ Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat history:
{history}

Latest user question:
{question}

Standalone version of the question:
"""
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
    return"\n\n".join( str(num+1)+') '+doc.page_content for num, doc in enumerate(docs))


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#    # | StrOutputParser()
# )#


for msg in msgs.messages:
    if msg.type == "ai" and hasattr(msg, "sources"):
        with st.chat_message("ai"):
            st.write(msg.content)
            expander = st.expander("See sources")
            expander.write(msg.sources)
    else:
        st.chat_message(msg.type).write(msg.content)

#question = st.text_input("Write a question about Gaia: ", key="input")
    
# def add_sources(docs):
#     lines = []
#     #lines.append('\nSources:')
#     for num, rd in enumerate(docs): #result["source_documents"]):
#         doc_info = []
#         doc_info.append(str(num+1)+') '+str(rd.metadata["title"]))
#         section_info = []
#         for item in rd.metadata:
#             if item.startswith('Header'):
#                 section_info.append(rd.metadata[item])
#         if rd.metadata.get("paragraph"):
#             section_info.append('paragraph: '+rd.metadata["paragraph"])
#         doc_info.append('   (Section: '+', '.join(section_info)+')')
#         doc_info.append('\n'+rd.metadata["link"])
#         lines.append(''.join(doc_info))
#     #text = '\"\"\"'+'\n'.join(lines)+'\"\"\"'
#     return '\n'.join(lines)    

############################################
#Refence handling

def add_sources(docs, source_numbers):
    lines = []
    #lines.append('\nSources:')
    if docs and source_numbers:
        print(source_numbers)
        for count, num in enumerate(source_numbers):
            rd = docs[int(num)-1]
            doc_info = []
            doc_info.append(str(count+1)+') '+str(rd.metadata["title"]))
            section_info = []
            for item in rd.metadata:
                if item.startswith('Header'):
                    section_info.append(rd.metadata[item])
            if section_info:
                doc_info.append('  \n   (Section: '+', '.join(section_info)+')')
            doc_info.append('  \n'+rd.metadata["link"])
            lines.append(''.join(doc_info))
    #text = '\"\"\"'+'\n'.join(lines)+'\"\"\"'
    else:
        lines = ["The information presented here does not explicitly reference the sources that were selected for the EcoTalkBot project. Extra caution with respect to accuracy may be in order."]
    return '  \n'.join(lines)

def replace_in_text(x, y, text):
    # Define the regex pattern to match 'Document x' with exact match on x
    pattern = rf'Document {x}(?=\b|\D)'
    # Define the replacement text 'Document y'
    replacement = f'Source {y}'
    # Use re.sub() to replace all instances of 'Document x' with 'Document y'
    updated_text = re.sub(pattern, replacement, text) 
    return updated_text

def replace_documents_list(text):
    # Define the regex pattern to match '(Documents x, y, z)'
    pattern = r'\(Documents? (\d+(?:, \d+)*(?:,? and \d+)?)\)'
    # Replacement function to reformat the matched text
    def replacement_function(match):
        # Extract the list of numbers from the match
        numbers = match.group(1)
        #print(numbers)
        number_list = re.split(r', and |, | and ', numbers)
        #print(number_list)
        # Join each number with 'Document ' prefix
        new_text = ', '.join([f'Document {num}' for num in number_list])
        #print(new_text)
        # Return the formatted text in the desired format
        return f'({new_text})'
    # Use re.sub() to replace all instances of '(Documents x, y, z)' with '(Document x, Document y, Document z)'
    updated_text = re.sub(pattern, replacement_function, text)
    return updated_text

def f7(seq): #deduplication of list while keeping order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
    
def used_sources(answer, lendocs):
    listed_pattern = r'\d+, ?\d'
    listed = re.findall(listed_pattern, answer)
    if listed:
        answer = replace_documents_list(answer)
    pattern = r'Document \d+'
    used = re.findall(pattern, answer)
    used = f7(used)
    used = [u.split()[-1] for u in used]
    remove = [u for u in used if int(u) > lendocs]
    used = [u for u in used if not u in remove]
    for num, u in enumerate(used):
        answer = replace_in_text(u, str(num+1), answer)
    if remove:
        for rn in remove:
            if '(Document '+rn+')' in answer:
                answer = answer.replace('(Document '+rn+')', '')
            elif 'Document '+rn+',' in answer:
                answer = answer.replace('Document '+rn+', ', '')
            elif ', Document '+rn in answer:
                answer = answer.replace(', Document '+rn, '')
    return answer, used
#########################################################

# def contains_referring(query):
#     words = [t.strip(',.:;?!\"\'') for t in query.split()]
#     for w in words:
#         if w.lower() in ['the', 'it', 'its', "it's", 'they', 'their', "they're", 'this', 'that' 'these', 'those', 'point', 'item', 'my', 'such', 'here', 'there', 'more', 'most', 'teh']:
#             return True
#         if w.isdigit():
#             return True

if user_input := st.chat_input():
    print(user_input)
    st.chat_message("human").write(user_input)
    prev_conv = '\n'.join([msg.type+': '+msg.content for msg in msgs.messages[-2:]])
    with st.spinner('Retrieving relevnat documents...'):
        contextualizing_prompt = contextualizing_template.format(history=prev_conv, question=user_input)
        print(contextualizing_prompt)
        contextualized_result = gpt3_5.invoke(contextualizing_prompt)
        vector_query = contextualized_result.content
        print(vector_query)
        retrieved = vectorstore.similarity_search(vector_query,k=20)
        cross_inp = [[vector_query, d.page_content] for d in retrieved]
        cross_scores = cross_encoder.predict(cross_inp)
        scored_pos = [(score, d) for score, d in zip(cross_scores, retrieved) if score > 0]
        if scored_pos:
            reranked = sorted(scored_pos, key=lambda tup: tup[0], reverse=True)
            docs = [r[1] for r in reranked[:7]]
        else:
            scored_neg = [(score, d) for score, d in zip(cross_scores, retrieved)]
            reranked = sorted(scored_neg, key=lambda tup: tup[0], reverse=True) #if there are no positive score, take the 2 least negative ones
            docs = [r[1] for r in reranked[:2]]
    with st.spinner('Generating response...'):
        full_prompt = template.format(context=format_docs(docs), question=user_input, conversation=prev_conv)
        print(full_prompt)
        result = gpt4.invoke(full_prompt)
        #sources = add_sources(docs)
        user_msg = BaseMessage(type="human", content=user_input)
        msgs.add_message(user_msg)
        ai_answer, source_numbers = used_sources(result.content, len(docs))
        print(ai_answer)
        sources = add_sources(docs, source_numbers)
    with st.chat_message("ai"):
        st.write(ai_answer)#+add_sources(docs))
        expander = st.expander("See sources")
        expander.write(sources) 
    ai_msg = BaseMessage(type="ai", content=ai_answer)
    setattr(ai_msg, 'sources', sources)
    msgs.add_message(ai_msg)    
    
   
    
    
# Check the result of the query

# Check the source document from where we 
# for rd in result["source_documents"]:
#     print(rd)
# print('\n')
# print(result["result"])