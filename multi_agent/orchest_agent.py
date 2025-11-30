from typing import Callable, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from search_agent import SearchResultResponse
from search_tools import prepare_search_tools
from utils import AgentConfig, NamedCallback, preferred_sites
from websearch_agent import ResearchReport, websearch_instructions
from websearch_tools import get_page_content as fetch_page_content, web_search as brave_web_search

SEARCH_TOOLS = prepare_search_tools()


class Reference(BaseModel):
    title: Optional[str] = None
    episode_name: Optional[str] = None
    authors: Optional[str] = None
    published_year: Optional[int] = None
    url: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

class Section(BaseModel):
    heading: str = ""
    content: str = ""

class ResultResponse(BaseModel):
    context: str = ""
    sections: List[Section] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)

    def format_response(self) -> str:
        output = "### Context\n"
        output += f"{self.context}\n\n"

        for section in self.sections:
            output += f"### {section.heading}\n\n"
            output += f"{section.content}\n\n"

        output += "### References\n"
        for ref in self.references:
            if ref.episode_name: 
                output += f"- {ref.episode_name} ({ref.start_time}-{ref.end_time})\n"
            else:
                output += f"- {ref.authors} ({ref.published_year}). *{ref.title}*. {ref.url}\n"
        return output.strip()


orchestrator_instructions = """
You are the Orchestrator in charge of delegating tasks to the right specialized agent. YOUR SOLE RESPONSIBILITY IS TO FIND THE RIGHT AGENT FOR THE JOB. 
Your agents are high specialized. Make sure they answer the questions. 

YOUR CHOICE OF AGENTS:
- clarifier_agent rewrites the original query in exactly 3 distinct ways which will be passed downstream to the search agent

- search_agent works with the elastic search vector store and has two tools:
   - embed: embed query strings
   - vector_search: search the vector store for matches between queries and the huberman knowledge base 

- websearch_agent searches the web for further information on research topics and has two tools:
   - search_web: find relevant links pertaining to research topics across a selected set of web domains.
   - web_page_content: retrieve the Markdown content of selected webpages.

YOUR WORKFLOW: 
- ALWAYS invoke the clarifier_agent first to rephrase the user query; this increases the chances of better search results
- Next, invoke the search_agent. The search_agent makes one search to the vector store using all three queries to increase chances of matches. Make sure search_agent comprehensively searches the vector database.
- Finally, the user may ask for more information. You ask the user if they want to proceed with further research, and ONLY THEN SHOULD YOU INVOKE THE websearch_agent.

IMPORTANT RULES:
- Web links you provide must be accurate and active. 
- BE RIGOROUS ABOUT CITATIONS.
- NEVER INVENT SOURCES, CITATIONS, TIME STAMPS, EPISODE NAMES. Always explicit state that when you are unsure of something.
- Never invent tool names.
- Never use external knowledge except what is returned by search_agent or websearch_agent. Explicitly state when you are unsure of something.

FOLLOW THIS OUTPUT STRUCTURE:
- Context: concise summary of what was done, rephrase the user question, and what will be in the output.
- Content section(s)
- References
""".strip()


clarifier_instructions = """
You assist the search_agent and websearch_agent.
You take a user's query and rewrite it 3 distinct ways using different phrasing, key terms, related subquestions.
""".strip()


search_instructions = """
You are an assistant that specializes in finding relevant passages from the Huberman Lab podcast archive (topics include but not limited to sleep, motivation, neuroscience, fitness, general health). 

SEARCH STRATEGY
- Do ONE vector search with embeddings from all queries.
- Merge the retrieved chunks, synthesize an answer based on the retrieved chunks in natural language, and cite every statement with its reference metadata. Make sure you include the rephrased question in your response.
- Paraphrase the user's query clearly and include this in your final response.
- If no relevant chunks are found after all rewrites, state that explicitly and offer general guidance.

TOOLS YOU CAN USE
- embed - embed all queries
- vector_search - fetch relevant chunks

IMPORTANT RULES
- Call vector search ONE time.
- Search comprehensively to provide a comprehensive response.
- IMPORTANT: Aim for more than one reference. DO NOT INVENT REFERENCES. BE RIGOROUS ABOUT CITATIONS. Explictly state some information is from the Huberman Lab podcast. 
- Use only information returned from the vector search tool; never invent facts. Explicitly state when you are giving general guidance.
- Rewrite the user's question clearly for each response and ensure you answer that question.
""".strip()


def create_orchestrator(config: Optional[AgentConfig] = None) -> Agent:
    config = config or AgentConfig()
    return Agent(
        name="orchestrator",
        instructions=orchestrator_instructions,
        model=config.model,
        output_type=ResultResponse,
    )


def create_clarifier_agent(config: Optional[AgentConfig] = None) -> Agent:
    config = config or AgentConfig()
    return Agent(
        name="clarifier_agent",
        instructions=clarifier_instructions,
        model=config.model,
        output_type=RewriteResponse,
    )


class RewriteResponse(BaseModel):
    rewrites: str


def create_search_agent(config: Optional[AgentConfig] = None) -> Agent:
    config = config or AgentConfig()
    return Agent(
        name="search_agent",
        instructions=search_instructions,
        model=config.model,
    )


def create_websearch_agent(config: Optional[AgentConfig] = None) -> Agent:
    config = config or AgentConfig()
    return Agent(
        name="websearch_agent",
        instructions=websearch_instructions,
        model=config.model,
        output_type=ResearchReport,
    )


def initialize_orchestrator(
    config: Optional[AgentConfig] = None,
    tool_event_getter: Optional[Callable[[], Optional[Callable]]] = None,
) -> tuple[Agent, NamedCallback]:
    orchestrator = create_orchestrator(config)
    clarifier_agent = create_clarifier_agent(config)
    search_agent = create_search_agent(config)
    websearch_agent = create_websearch_agent(config)

    def make_callback(agent):
        return NamedCallback(agent, observer_getter=tool_event_getter)

    @websearch_agent.tool
    async def web_search_tool(ctx: RunContext, query: str):
        """Fetch candidate URLs from Brave Search limited to trusted domains."""
        results = brave_web_search(query, preferred_sites)
        return results or []

    @websearch_agent.tool
    async def get_page_content_tool(ctx: RunContext, url: str):
        """Retrieve Markdown content for a given URL using the reader proxy."""
        content = fetch_page_content(url)
        return content or ""

    @orchestrator.tool
    async def rewrite_user_query(ctx: RunContext, query: str) -> str:
        callback = make_callback(clarifier_agent)
        results = await clarifier_agent.run(
            user_prompt=query,
            event_stream_handler=callback,
        )
        return results.output

    @orchestrator.tool
    async def embed(ctx: RunContext, query: str, config: Optional[AgentConfig] = None):
        del config  # Search agent tools own the embedding configuration.
        return SEARCH_TOOLS.embedding(query)

    @orchestrator.tool
    async def vector_search(ctx: RunContext, query: str, config: Optional[AgentConfig] = None):
        cfg = config or AgentConfig()
        prior_outputs = []
        for message in ctx.messages:
            for part in message.parts:
                if part.part_kind == "tool-return" and part.tool_name == "embed":
                    prior_outputs.append(part.content)

        prior_text = "".join(str(x) for x in prior_outputs)
        prompt = f"""
        User query:
        {query}

        Prior clarification:
        {prior_text}
        """.strip()

        callback = make_callback(search_agent)
        results = await search_agent.run(
            user_prompt=prompt,
            event_stream_handler=callback,
            output_type=SearchResultResponse,
        )
        return results.output.model_dump()

    @orchestrator.tool
    async def search_web(ctx: RunContext, query: str):
        prior_outputs = []
        for message in ctx.messages:
            for part in message.parts:
                if part.part_kind == "tool-return" and part.tool_name == "rewrite_user_query":
                    prior_outputs.append(part.content)

        prior_text = "\n".join(str(x) for x in prior_outputs)
        prompt = f"""
        User query:
        {query}

        Prior clarification:
        {prior_text}
        """.strip()

        callback = make_callback(websearch_agent)
        results = await websearch_agent.run(
            user_prompt=prompt,
            event_stream_handler=callback,
        )
        return results.output

    @orchestrator.tool
    async def web_page_content(ctx: RunContext, url: str, query: str, config: Optional[AgentConfig] = None):
        content = fetch_page_content(url)
        if not content:
            print(f"Error fetching content from {url}")

        prior_outputs = []
        for message in ctx.messages:
            for part in message.parts:
                if part.part_kind == "tool-return" and part.tool_name == "search_web":
                    prior_outputs.append(part.content)

        prior_text = "\n".join(str(x) for x in prior_outputs)
        prompt = f"""
        User query:
        {query}

        Prior clarification:
        {prior_text}
        """.strip()

        callback = make_callback(websearch_agent)
        results = await websearch_agent.run(
            user_prompt=prompt,
            event_stream_handler=callback,
            output_type=ResearchReport,
        )
        return results.output.format_response()

    callback = make_callback(orchestrator)
    return orchestrator, callback
