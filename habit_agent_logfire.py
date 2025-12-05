from research_qdrant import create_research_agent
from tools.utils import NamedCallback
import asyncio

from jaxn import JSONParserHandler, StreamingJSONParser
import logfire
from agent_logging.agent_logging import log_streamed_run, save_log
from typing import Any, Dict


logfire.configure()
logfire.instrument_pydantic_ai()

class FinalReportHandler(JSONParserHandler):
    
    def on_field_end(self, path: str, field_name: str, value: str, parsed_value: Any = None) -> None:
        if field_name == "description":
            print(f'### Description')
            print(f'{value}')
        
        if field_name == "heading":
            print(f'\n\n## {value}\n')
        
        
         
    def on_value_chunk(self, path: str, field_name: str, chunk: str) -> None:
        if field_name == "content":
            print(chunk, end="", flush=True)
    
    def on_array_item_end(self, path: str, field_name: str, item: Dict[str, Any] = None) -> None:
       if field_name == "references":
            print(f"\n ### References")
            if "episode_name" in item:
                print(f' - {item["episode_name"]} ({item["start"]} - {item["end"]})')
            if "url" in item:
                title = item.get("title", "Untitled")
                url = item.get("url", "")
                print(f'  - *{title}* {url}')
       


async def main():

    user_prompt = "What is the association between coffee and alzheimer's?"
    agent = create_research_agent()
    agent_callback = NamedCallback(agent)

    handler = FinalReportHandler()

    parser = StreamingJSONParser(handler)

    previous_text = ""

    async with agent.run_stream(user_prompt, event_stream_handler=agent_callback) as result:
        async for item, last in result.stream_responses(debounce_by=0.01):
            for part in item.parts:
                if not hasattr(part, "tool_name"):
                    continue
                if part.tool_name != "final_result":
                    continue

                current_text = part.args
                delta = current_text[len(previous_text):]
                parser.parse_incremental(delta)
                previous_text = current_text
        log_entry = await log_streamed_run(agent, result)
        save_log(log_entry)

if __name__ == "__main__":
    asyncio.run(main())