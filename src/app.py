from pprint import pprint
from graph import get_workflow
from vector_store import build_vector_store
from generator import get_response_generator
from graders import get_answer_grader, get_hallucination_grader, get_relevance_grader
from router import get_question_router
from tools import get_web_search_tool


def main():
    collection_name = "rag-chroma"
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    retriever = build_vector_store(collection_name, urls).as_retriever()
    rag_chain = get_response_generator()
    retrieval_grader = get_relevance_grader()
    web_search_tool = get_web_search_tool()
    question_router = get_question_router()
    hallucination_grader = get_hallucination_grader()
    answer_grader = get_answer_grader()
    workflow = get_workflow(
        retriever=retriever,
        rag_chain=rag_chain,
        retrieval_grader=retrieval_grader,
        web_search_tool=web_search_tool,
        question_router=question_router,
        hallucination_grader=hallucination_grader,
        answer_grader=answer_grader,
    )

    app = workflow.compile()

    inputs = {"question": "What are the types of agent memory?"}
    inputs_iterator = app.stream(inputs)
    for output in inputs_iterator:
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])


if __name__ == "__main__":
    main()
