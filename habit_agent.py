from research_qdrant import create_research_agent
from tools.utils import NamedCallback
import asyncio


async def main():

    user_prompt = "What is the association between coffee and alzheimer's?"
    agent = create_research_agent()
    agent_callback = NamedCallback(agent)

    result = await agent.run(user_prompt, event_stream_handler=agent_callback)
    response = result.output

    print(response.format_response())


if __name__ == "__main__":
    asyncio.run(main())