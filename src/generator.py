from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

from config import local_llm


def get_response_generator():
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
      Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
      Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
      Question: {question} 
      Context: {context} 
      Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    llm = ChatOllama(model=local_llm, temperature=0)

    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain
