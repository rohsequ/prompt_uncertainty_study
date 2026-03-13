import argparse
import inspect
import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field

from agentdojo.task_suite.load_suites import get_suite, get_suites
from utils.llm_utils import LLMConfig, get_llm_provider


logging.basicConfig(level=logging.INFO)

# ============================================================================
# Prompt Formats
# ============================================================================

RATING_SYSTEM_PROMPT = """You are an expert at analyzing instructions given to an autonomous AI agent.
Your goal is to evaluate the 'ambiguity' of a user's prompt based on how clearly it defines the task and constraints, compared to the actual expected verification logic and ground truth tools.

Rate the user's prompt on a scale of 1 to 5, where:
1 - Extremely Ambiguous: Very vague, lacks specific details, relies heavily on agent context/assumption. The agent has to guess the intent or parameters.
2 - Highly Ambiguous: Missing some key constraints but provides the general goal. The agent might need to explore to find exact details.
3 - Moderately Ambiguous: Provides necessary constraints but leaves some execution details open or unspecified.
4 - Slightly Ambiguous: Clear and direct, includes almost all necessary parameters.
5 - Extremely Clear: Overly explicit, often mentioning exact tool names, arguments, and step-by-step instructions.

You will be provided with:
1. The ORIGINAL USER PROMPT.
2. The GROUND TRUTH functionality (the ideal tool calls).
3. The UTILITY VERIFICATION logic (how the environment checks if the agent succeeded).

Respond with a JSON object containing a "reasoning" string and an "ambiguity_score" integer (1-5).
"""


class RatingResponse(BaseModel):
    reasoning: str = Field(description="Reasoning for the ambiguity score.")
    ambiguity_score: int = Field(description="Integer score from 1 to 5.")


GENERATION_SYSTEM_PROMPT = """You are an expert at generating instructions for autonomous AI agents.
You will be provided with an original user prompt, its current ambiguity score (1-5), and the ground truth/verification logic for a task.

Your goal is to generate ALL 5 variations of this prompt corresponding to the 1-5 ambiguity scale.
1 - Extremely Ambiguous: Very vague, lacks specific details, relies heavily on agent context/assumption.
2 - Highly Ambiguous: Missing some key constraints but provides the general goal.
3 - Moderately Ambiguous: Provides necessary constraints but leaves some execution details open.
4 - Slightly Ambiguous: Clear and direct, includes almost all necessary parameters.
5 - Extremely Clear: Overly explicit, often mentioning exact tool names, arguments, and step-by-step instructions.

Make sure the prompts are natural and read like something a real human user would ask the agent.
You MUST output a JSON object containing exactly 5 keys: "L1", "L2", "L3", "L4", "L5", each mapping to the prompt string for that level.
"""


class GenerationResponse(BaseModel):
    L1: str
    L2: str
    L3: str
    L4: str
    L5: str


# ============================================================================
# Main Logic
# ============================================================================


def assess_ambiguity(
    prompt: str, ground_truth_code: str, utility_code: str, llm_config: LLMConfig
) -> RatingResponse:
    user_message = f"""
ORIGINAL USER PROMPT:
{prompt}

GROUND TRUTH CODE:
{ground_truth_code}

UTILITY VERIFICATION CODE:
{utility_code}

Return ONLY a JSON object matching the requested schema.
"""

    # Adding a system prompt nudge to output JSON
    enhanced_system = RATING_SYSTEM_PROMPT + "\n\nYou MUST respond with valid JSON."

    provider = get_llm_provider(llm_config)
    messages = [
        {"role": "system", "content": enhanced_system},
        {"role": "user", "content": user_message},
    ]
    response_str = provider.invoke(messages, response_schema=RatingResponse).content

    try:
        # Simple extraction if the model wrapped it in markdown code blocks
        if "```json" in response_str:
            response_str = response_str.split("```json")[1].split("```")[0].strip()
        elif "```" in response_str:
            response_str = response_str.split("```")[1].strip()

        data = json.loads(response_str)
        return RatingResponse(**data)
    except Exception as e:
        logging.error(f"Failed to parse rating JSON: {response_str}\nError: {e}")
        # Default fallback
        return RatingResponse(reasoning="Failed to parse", ambiguity_score=3)


def generate_variations(
    prompt: str,
    score: int,
    ground_truth_code: str,
    utility_code: str,
    llm_config: LLMConfig,
) -> GenerationResponse:
    user_message = f"""
ORIGINAL USER PROMPT (Currently rated as Level {score}):
{prompt}

GROUND TRUTH CODE:
{ground_truth_code}

UTILITY VERIFICATION CODE:
{utility_code}

Generate all 5 variations (L1 to L5) as a JSON object.
"""
    enhanced_system = (
        GENERATION_SYSTEM_PROMPT
        + "\n\nYou MUST respond with valid JSON containing keys L1, L2, L3, L4, L5."
    )

    provider = get_llm_provider(llm_config)
    messages = [
        {"role": "system", "content": enhanced_system},
        {"role": "user", "content": user_message},
    ]
    response_str = provider.invoke(messages, response_schema=GenerationResponse).content

    try:
        if "```json" in response_str:
            response_str = response_str.split("```json")[1].split("```")[0].strip()
        elif "```" in response_str:
            response_str = response_str.split("```")[1].strip()

        data = json.loads(response_str)
        return GenerationResponse(**data)
    except Exception as e:
        logging.error(f"Failed to parse generation JSON: {response_str}\nError: {e}")
        # Default fallback
        return GenerationResponse(L1=prompt, L2=prompt, L3=prompt, L4=prompt, L5=prompt)


def generate_prompts(
    agent: str, provider: str, model: str, output_dir: Path, limit: int | None = None
):
    agent_output_dir = output_dir / agent
    agent_output_dir.mkdir(parents=True, exist_ok=True)

    # Needs a higher token count for generating 5 variations, defaulting to None max output
    llm_config = LLMConfig(provider=provider, model=model)

    logging.info(f"Loading suite: {agent}")
    benchmark_version = "v1.2.2"  # Matches your existing run_experiments.py
    suite = get_suite(benchmark_version, agent)

    tasks_to_process = list(suite.user_tasks.items())
    if limit:
        tasks_to_process = tasks_to_process[:limit]

    for task_name, task_obj in tasks_to_process:
        logging.info(f"Processing task: {task_name}")

        # 1. Extract Info
        prompt = task_obj.PROMPT

        try:
            ground_truth_code = inspect.getsource(task_obj.ground_truth)
        except Exception:
            ground_truth_code = "No ground truth definition found."

        try:
            utility_code = inspect.getsource(task_obj.utility)
        except Exception:
            utility_code = "No utility definition found."

        # 2. Assess Ambiguity
        logging.info(f"  Assessing original prompt...")
        rating = assess_ambiguity(prompt, ground_truth_code, utility_code, llm_config)
        logging.info(
            f"  Rated as L{rating.ambiguity_score}: {rating.reasoning[:100]}..."
        )

        # 3. Generate Variations
        logging.info(f"  Generating variations L1-L5...")
        variations = generate_variations(
            prompt, rating.ambiguity_score, ground_truth_code, utility_code, llm_config
        )

        # 4. Save Output
        output_data = {
            "task_id": task_name,
            "original_prompt": prompt,
            "original_assessed_score": rating.ambiguity_score,
            "assessment_reasoning": rating.reasoning,
            "variations": variations.model_dump(),
        }

        out_file = agent_output_dir / f"{agent}_{task_name}.json"
        with open(out_file, "w") as f:
            json.dump(output_data, f, indent=4)

        logging.info(f"  Saved to {out_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ambiguous prompt variations for AgentDojo tasks."
    )
    parser.add_argument(
        "--agent",
        nargs="+",
        default=["workspace"],
        help="List of agent suites to load (e.g. workspace, banking) or 'all' for every available suite",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["openai", "ollama", "ollama-cloud", "deepinfra"],
        help="LLM Provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:8b",
        help="Model name (e.g. gpt-4o for openai, qwen3:8b for ollama)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ambiguity_prompts"),
        help="Output directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only a subset of tasks for testing. Default 0 (process all tasks).",
    )
    args = parser.parse_args()

    agents = args.agent
    benchmark_version = "v1.2.2"
    if "all" in agents:
        agents = list(get_suites(benchmark_version).keys())

    limit = None if args.limit == 0 else args.limit

    for agent in agents:
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting Prompt Generation for Agent: {agent}")
        logging.info(f"{'='*50}")
        try:
            generate_prompts(
                agent=agent,
                provider=args.provider,
                model=args.model,
                output_dir=args.output_dir,
                limit=limit,
            )
        except Exception as e:
            logging.error(f"Failed to generate for suite {agent}: {e}")


if __name__ == "__main__":
    main()
