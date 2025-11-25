from tools.search_agent import SearchResultResponse
import main

def test_references():
    user_prompt = "is mushroom coffee better than ordinary coffee"
    result = main.run_agent_sync(user_prompt=user_prompt)

    response: SearchResultResponse = result.output
    print(response.format_response())

    assert all(section.references for section in response.sections), "Expecting at least one reference in each section."