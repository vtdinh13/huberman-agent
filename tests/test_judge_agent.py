import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai import Agent, AgentRunResult
from pydantic import BaseModel

from tests.utils import get_tool_calls
import main



judge_instructions = """
You are an expert and impartial judge. Your sole task is to judge the quality of responses given from an AI agent. Make your judgement based only on the criteria provided. 

RULES
- Be neutral and consistent.
- Do not add new facts or reinterpret the user query beyond what is given.
- Provide justification to your judgement, grounded in text of the response.
- DO NOT INJECT STYLISTIC OR PERSONAL PERFERENCE judgements.
- Do not attempt to fix the responses; evaluate only.
- Give response as a pass or fail.
- Provide summary feedback. 
""".strip()


class JudgeCriterion(BaseModel):
    criterion_description: str
    passed: bool
    judgement: str


class JudgeFeedback(BaseModel):
    criteria: list[JudgeCriterion]
    feedback: str


def create_judge():
    judge = Agent(
        name="judge",
        instructions=judge_instructions,
        model="openai:gpt-5-nano",
        output_type=JudgeFeedback,
    )
    return judge
async def evaluate_agent_performance( criteria: list[str], result: AgentRunResult, output_transformer: callable = None
) -> JudgeFeedback:
    judge = create_judge()

    tool_calls = get_tool_calls(result)

    output = result.output
    if output_transformer is not None:
        output = output_transformer(output)

    criteria_str = "\n".join(criteria)
    tool_calls_str = "\n".join(str(c) for c in tool_calls)

    user_prompt = f"""
    Evaluate the agent's performance based on the following criteria:
    <CRITERIA>
    {criteria_str}
    </CRITERIA>

    The agent's final output was:
    <AGENT_OUTPUT>
    {output}
    </AGENT_OUTPUT>

    Tool calls:
    <TOOL_CALLS>
    {tool_calls_str}
    </TOOL_CALLS>
    """

    print("Judge evaluating with prompt:")
    print("-----")
    print(user_prompt)
    print("-----")

    eval_results = await judge.run(
        user_prompt=user_prompt
    )

    return eval_results.output
def serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    raise TypeError(f"Type {type(obj)} not serializable")

TEST_RESULTS_DIR = Path(__file__).parent / "test_results"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_PATH = TEST_RESULTS_DIR / "judge_eval_results.json"

def append_eval_record(record: dict) -> None:
    """Persist evaluation output so each test run is captured."""
    if EVAL_RESULTS_PATH.exists():
        try:
            existing = json.loads(EVAL_RESULTS_PATH.read_text())
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []

    existing.append(record)

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f_out:
        json.dump(existing, f_out, indent=2, default=serializer)

@pytest.mark.asyncio
async def test_eval_agent():
    user_prompt = "what is better, match or coffee"

    result = await main.run_agent(user_prompt)

    print(result.output)
    
    criteria = [
        "Did the agent follow the user's instructions?",
        "Is the answer clear and correct?",
        "Did the response fully address the query?",
        "Did the response cover all key aspects of the request?",
    ]

    judge_feedback = await evaluate_agent_performance(
        criteria,
        result,
        output_transformer=lambda output: output
    )

    print(judge_feedback)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_prompt": user_prompt,
        "agent_output": result.output,
        "criteria": criteria,
        "tool_calls": [str(call) for call in get_tool_calls(result)],
        "judge_feedback": judge_feedback.model_dump(),
    }
    append_eval_record(record)

    for criterion in judge_feedback.criteria:
        assert criterion.passed, f"Criterion failed: {criterion.criterion_description}, {criterion.judgement}"
