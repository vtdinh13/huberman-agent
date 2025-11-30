from pydantic import BaseModel, Field
from pydantic_ai import Agent

from websearch_tools import get_page_content, web_search

from typing import Optional
from utils import AgentConfig


websearch_instructions = """
You are a focused research assistant specializing in analyzing complex or esoteric scientific and technical articles.
Your job is to search the web using the Brave API, retrieve content, analyze it accurately, and synthesize insights.

YOUR TASKS:
- Overview  
   - Briefly describe what you searched for, what pages you retrieved, and what your final output will include.

- Article Summaries 
   - Provide a detailed but concise summary of all articles in one section. 
   - Explain key arguments, evidence, findings, methods, and conclusions.  
   - Include strengths, limitations, assumptions, and context.  
   - ALWAYS cite the source and include the URL.

- Synthesis Across All Articles  
   - Compare and contrast the articles.  
   - Identify areas of agreement, disagreement, novelty, or emerging themes.  
   - Evaluate strengths and weaknesses; highlight pros, cons, and methodological robustness.  
   - Extract core insights and patterns supported by the cited content.  
   - Organize insights into clear topical clusters.

- Final Key Findings  
   - Provide a conclusion of the most important insights across all sources.  
   - Every claim MUST be traceable to a cited article.

RULES (MANDATORY):
- BE RIGOROUS ABOUT CITATIONS. Web links MUST be active, valid, and accurate. NEVER invent web links. Explictly state what you don't know.
- ALWAYS INCLUDE REFERENCES -- author names, publishing year, urls -- for every fact, claim, or summary. If the name of the author is missing, include the name of the journal instead.
- NEVER fabricate or speculate. If information is missing, explicitly state that it is not available.  
- Use ONLY content from the retrieved webpages.  
- Prioritize clarity, accuracy, and conciseness.  
- Do not rely on prior knowledge—everything must come from the pages you fetched.
- ALWAYS VERIFY AND VALIDATE that your text is valid JSON. If unsure, reformat before completing your task.

AVAILABLE TOOLS:
- web_search: search a list of specified websites for matching pages.  
- get_page_content: retrieve the Markdown content of selected webpages.
"""


class Reference(BaseModel):
    title: str
    authors: str
    published_year: Optional[int] = None
    url: str 

    def format_citation(self) -> str:
        citation = f"{self.authors} ({self.published_year}). *{self.title}*. {self.url}"
        return citation


class Section(BaseModel):
    heading: str
    content: str
    references: list[Reference] = Field(default_factory=list)


class Conclusion(BaseModel):
    content: str


class ResearchReport(BaseModel):
    overview: str
    sections: list[Section]
    conclusion: Conclusion
    references: list[Reference] = Field(default_factory=list)

    def format_response(self) -> str:
        output = "### Overview \n"
        output += f"{self.overview}\n"

        for section in self.sections:
            output += f"### {section.heading}\n"
            output += f" {section.content}\n"
            if section.references:
                output += "#### References \n"
                for idx, ref in enumerate(section.references, start=1):
                    output += f"{idx}. {ref.format_citation()}\n\n"

        output += f" ### Conclusion\n"
        output += f" {self.conclusion.content}\n"

    
        return output.strip()


# def create_websearch_agent():
#     return Agent(
#         name="websearch_agent",
#         instructions=websearch_instructions,
#         model="openai:gpt-4o-mini",
#         tools=[web_search, get_page_content],
#         output_type=ResearchReport,
#     )

def create_websearch_agent(config: Optional[AgentConfig] = None) -> Agent:
    """Create the websearch agent that uses Brave API tools."""
    if config is None:
        config = AgentConfig()

    websearch_agent = Agent(
        name="websearch_agent",
        instructions=websearch_instructions,
        model=config.model)
    
    return websearch_agent