from habit_agent import create_research_agent
from tools.utils import NamedCallback
import asyncio


agent = create_research_agent()
agent_callback = NamedCallback(agent)

async def run_agent(user_prompt:str):
    result = await agent.run(user_prompt, event_stream_handler=agent_callback)
    return result


def run_agent_sync(user_prompt:str):
    return asyncio.run(run_agent(user_prompt))