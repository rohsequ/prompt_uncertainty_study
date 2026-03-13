import argparse
import logging
from pathlib import Path

# Important: Make sure all attacks are registered
import agentdojo.attacks  # noqa: F401
import agentdojo.models as models
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import (
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.logging import OutputLogger

logging.basicConfig(level=logging.INFO)

BENCHMARK_VERSION = "v1.2.2"


def main():
    parser = argparse.ArgumentParser(
        description="Run AgentDojo benchmarks with TransformersLLM."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the local HF model. e.g. /research-storage/lab/rohseque/huggingface_models/meta-llama-3-1-8b-instruct",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["banking", "travel", "workspace", "slack"],
        help="List of agent types to test.",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["none", "direct"],
        help="List of attacks to test. 'none' will run without injections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Output directory for benchmark logs.",
    )
    parser.add_argument(
        "--limit-user-tasks",
        type=int,
        default=1,
        help="Number of user tasks to test (for quick testing).",
    )
    parser.add_argument(
        "--limit-injection-tasks",
        type=int,
        default=1,
        help="Number of injection tasks to test.",
    )

    args = parser.parse_args()

    from utils.pipeline_utils import setup_agentdojo_pipeline

    pipeline = setup_agentdojo_pipeline(args.model_path)

    with OutputLogger(logdir=str(args.output_dir)):
        for agent_type in args.agents:
            logging.info(f"Loading suite: {agent_type} (version {BENCHMARK_VERSION})")
            suite = get_suite(BENCHMARK_VERSION, agent_type)

            # Get subset of tasks for quick testing
            user_tasks = (
                list(suite.user_tasks.keys())[: args.limit_user_tasks]
                if args.limit_user_tasks
                else list(suite.user_tasks.keys())
            )
            injection_tasks = (
                list(suite.injection_tasks.keys())[: args.limit_injection_tasks]
                if args.limit_injection_tasks
                else list(suite.injection_tasks.keys())
            )

            for attack_name in args.attacks:
                logging.info(f"Running {agent_type} suite with attack: {attack_name}")
                if attack_name == "none":
                    results = benchmark_suite_without_injections(
                        agent_pipeline=pipeline,
                        suite=suite,
                        logdir=args.output_dir,
                        force_rerun=True,  # Set false to read from log cache
                        user_tasks=user_tasks,
                        benchmark_version=BENCHMARK_VERSION,
                    )
                else:
                    attack = load_attack(attack_name, suite, pipeline)
                    results = benchmark_suite_with_injections(
                        agent_pipeline=pipeline,
                        suite=suite,
                        attack=attack,
                        logdir=args.output_dir,
                        force_rerun=True,  # Set false to read from log cache
                        user_tasks=user_tasks,
                        injection_tasks=injection_tasks,
                        benchmark_version=BENCHMARK_VERSION,
                    )

                utils = results["utility_results"]
                secs = results["security_results"]

                # Simple summary
                total = len(utils)
                util_success = sum(utils.values()) if utils else 0
                sec_success = sum(secs.values()) if secs else 0

                print(f"--- Results for Agent: {agent_type}, Attack: {attack_name} ---")
                print(f"Total Task Configurations Run: {total}")
                print(f"Utility Success Rate: {util_success}/{total}")
                if attack_name != "none":
                    print(f"Security Success Rate: {sec_success}/{total}")
                print("-" * 50)


if __name__ == "__main__":
    main()
