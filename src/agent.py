from langchain_core.retrievers import RetrieverLike
from pprint import pprint
from typing import List
from graph import get_workflow
from vector_store import build_vector_store
from generator import get_response_generator
from graders import get_answer_grader, get_hallucination_grader, get_relevance_grader
from router import get_question_router
from tools import get_web_search_tool


class Agent:
    def __init__(self):
        self.app = None
        self.retriever: RetrieverLike = None
        self.current_collection_name: str = None
        self.current_collection_scope: str = None
        self.current_urls: List[str] = None

    def generate_response(
        self,
        question: str,
        collection_name: str,
        collection_scope: str,
        urls: List[str],
    ):
        if any(
            [
                self.app is None,
                self.current_collection_name != collection_name,
                self.current_collection_scope != collection_scope,
                self.current_urls != urls,
                self.retriever is None,
            ]
        ):
            self.app = self.get_app(collection_name, collection_scope, urls)

        inputs = {"question": question}
        inputs_iterator = self.app.stream(inputs)

        for output in inputs_iterator:
            for key, value in output.items():
                pprint(f"Finished running: {key}:")

        return value["generation"]

    def get_app(self, collection_name: str, collection_scope: str, urls: List[str]):
        self.retriever = build_vector_store(collection_name, urls).as_retriever()
        self.current_collection_name = collection_name
        self.current_collection_scope = collection_scope
        self.current_urls = urls

        rag_chain = get_response_generator()
        retrieval_grader = get_relevance_grader()
        web_search_tool = get_web_search_tool()
        question_router = get_question_router(collection_scope)
        hallucination_grader = get_hallucination_grader()
        answer_grader = get_answer_grader()
        workflow = get_workflow(
            retriever=self.retriever,
            rag_chain=rag_chain,
            retrieval_grader=retrieval_grader,
            web_search_tool=web_search_tool,
            question_router=question_router,
            hallucination_grader=hallucination_grader,
            answer_grader=answer_grader,
            collection_scope=collection_scope,
        )

        self.app = workflow.compile()

        return self.app


agent = Agent()
