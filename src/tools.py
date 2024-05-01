from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun


def get_web_search_tool():
    return DuckDuckGoSearchRun()
