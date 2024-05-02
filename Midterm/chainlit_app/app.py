from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from getpass import getpass
import openai
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import chainlit as cl
from dotenv import load_dotenv
load_dotenv()

loader = PyMuPDFLoader("meta_10k.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4250, chunk_overlap = 100)
texts = text_splitter.split_documents(documents)

embeddings_model = OpenAIEmbeddings(model = "text-embedding-3-small")
llm = ChatOpenAI(model = 'gpt-3.5-turbo', temperature=0)

qdrant = Qdrant.from_documents(texts, embeddings_model, location=":memory:", collection_name = '10K_RAG', force_recreate=True)
qdrant_retriever = qdrant.as_retriever()

metadata_field_info = [
]

document_content_desc = "Form 10-K annual report required by the U.S. Securities and Exchange Commission (SEC), that gives a comprehensive summary of a company's financial performance for company Meta for year 2023"

self_query_retriever = SelfQueryRetriever.from_llm(llm, qdrant, document_content_desc, metadata_field_info)

template = """You are an helpful assistant for question-answering tasks, specifically you are an expert in answering SEC 10-K report questions.
If you  don't know the answer, just say that you don't know.
Answer based on the context given to you, for a given question.
Always adhere to instructions given in the question.

Context:
{context}

Question:
{question}

Answer:"""

rag_chat_prompt = ChatPromptTemplate.from_template(template)

rag_qa_chain_sqr =  (
    {"context": itemgetter("question") | self_query_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_chat_prompt | llm, "context": itemgetter("context")}
)

@cl.on_chat_start
def start_chat():
    cl.user_session.set("chain", rag_qa_chain_sqr)
    
@cl.on_message
async def main(message: cl.message):
    chain = cl.user_session.get("chain")
    
    result = rag_qa_chain_sqr.invoke({"question": message.content})
    answer = result['response'].content
    source_documents = result['context']
    
    page_elements = []
    pages_all = []
    if source_documents:
        for i in range(len(source_documents)):
            source_page = f"{source_documents[i].metadata['page']}"
            pages_all.append(source_page)
        pages_all = ', '.join(pages_all)
        page_elements.append(cl.Text(content=pages_all, name="Source Pages", display="inline"))
    
    await cl.Message(content=answer, elements = page_elements).send()