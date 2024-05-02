### Router

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


from config import local_llm


def get_question_router(collection_scope: str):
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    prompt = PromptTemplate(
        template=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
      user question to a vectorstore or web search. Use the vectorstore for questions on {collection_scope}. You do not need to be stringent with the keywords 
      in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
      or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
      no premable or explaination. Question to route: {{question}} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()

    return question_router
