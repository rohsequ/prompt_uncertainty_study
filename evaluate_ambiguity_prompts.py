import argparse
import json
import logging
from pathlib import Path
from copy import deepcopy
import datetime
import csv
import os
import sys

from agentdojo.benchmark import run_task_without_injection_tasks
from agentdojo.task_suite.load_suites import get_suite

from utils.pipeline_utils import setup_agentdojo_pipeline
from utils.custom_logger import FullOutputLogger

logging.basicConfig(level=logging.INFO)

BENCHMARK_VERSION = "v1.2.2"


def evaluate_task_variations(
    suite,
    pipeline,
    task_json_path: Path,
    output_dir: Path,
    n_runs: int,
    csv_writer,
    model_name: str,
):
    """
    Evaluates all L1-L5 prompt variations defined in a task JSON file n_runs times.
    """
    with open(task_json_path, "r") as f:
        data = json.load(f)

    task_id = data["task_id"]
    variations = data.get("variations", {})

    if task_id not in suite.user_tasks:
        logging.warning(f"Task {task_id} not found in the suite. Skipping.")
        return

    original_task = suite.user_tasks[task_id]

    print(f"\n{'='*50}")
    print(f"Evaluating Task: {task_id}")
    print(
        f"Original Prompt (Rated {data['original_assessed_score']}): {data['original_prompt']}"
    )
    print(f"{'='*50}")

    results = {}

    with FullOutputLogger(logdir=str(output_dir)):
        for level, new_prompt in sorted(variations.items()):
            print(f"\n--- Testing Level: {level} ---")
            print(f"Prompt: {new_prompt}")
            results[level] = {}

            for run_idx in range(n_runs):
                # Deepcopy to avoid modifying the base suite task state for other tests if we ran in batch
                task_copy = deepcopy(original_task)

                # Override properties for this ambiguity level run
                task_copy.ID = f"{task_id}_{level}_run{run_idx}"
                task_copy.PROMPT = new_prompt

                # Run the agentdojo evaluation loop
                utility, security = run_task_without_injection_tasks(
                    suite=suite,
                    agent_pipeline=pipeline,
                    task=task_copy,
                    logdir=output_dir,
                    force_rerun=True,
                    benchmark_version=BENCHMARK_VERSION,
                )

                results[level][run_idx] = utility
                print(
                    f"[{level} | Run {run_idx}] Result - Utility (Task Succeeded): {utility}"
                )

                # Write to CSV tracking artifact
                csv_writer.writerow([model_name, task_id, level, run_idx, int(utility)])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Ambiguity Prompts on AgentDojo"
    )
    parser.add_argument(
        "--model-path", nargs="+", required=True, help="List of model paths"
    )
    parser.add_argument(
        "--agent", type=str, default="workspace", help="Agent suite to use"
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path("ambiguity_prompts"),
        help="Directory containing generated L1-L5 JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs_eval"),
        help="Output directory for logs",
    )
    parser.add_argument(
        "--provider",
        nargs="+",
        default=["transformers"],
        help="List of LLM Providers. Default: transformers",
    )
    parser.add_argument(
        "--gen-provider",
        type=str,
        default="openai",
        help="LLM Provider for prompt generation. Default: openai",
    )
    parser.add_argument(
        "--gen-model",
        type=str,
        default="gpt-4o-mini",
        help="Model string to use for prompt generation. Default: gpt-4o-mini",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=0,
        help="Number of tasks to limit testing to. Set to 0 to run all available tasks in the agent suite.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of times to run each prompt level.",
    )
    args = parser.parse_args()

    model_paths = (
        args.model_path if isinstance(args.model_path, list) else [args.model_path]
    )
    providers = args.provider if isinstance(args.provider, list) else [args.provider]

    if len(providers) == 1 and len(model_paths) > 1:
        providers = providers * len(model_paths)
    elif len(providers) != len(model_paths):
        logging.error(
            "Number of providers must match number of models, or exactly 1 provider must be given."
        )
        sys.exit(1)

    # Base log dir: results_TIME
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_dir / f"results_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    csv_file_path = base_output_dir / "results.csv"

    # Suite Setup
    logging.info(f"Loading suite: {args.agent}")
    suite = get_suite(BENCHMARK_VERSION, args.agent)
    logging.info(
        f"Total user tasks available in suite '{args.agent}': {len(suite.user_tasks)}"
    )

    # Find prompt variation JSONs
    agent_prompts_dir = args.prompts_dir / args.agent
    json_files = list(agent_prompts_dir.glob(f"{args.agent}_*.json"))

    n_tasks_to_eval = args.n_tasks if args.n_tasks > 0 else len(suite.user_tasks)

    # If we need more JSONs than what we have, generate them natively!
    if len(json_files) < n_tasks_to_eval:
        logging.info(
            f"Only {len(json_files)} tasks found, but {n_tasks_to_eval} needed. Automatically generating more natively..."
        )
        from generate_ambiguity_prompts import generate_prompts

        generate_prompts(
            agent=args.agent,
            provider=args.gen_provider,
            model=args.gen_model,
            limit=n_tasks_to_eval,
            output_dir=args.prompts_dir,
        )

        # Reload JSON files after generation
        json_files = list(agent_prompts_dir.glob(f"{args.agent}_*.json"))

    if not json_files:
        logging.warning(
            f"No JSON files found in {args.prompts_dir} matching {args.agent}_*.json"
        )
        return

    logging.info(f"Found {len(json_files)} task evaluation files.")

    if args.n_tasks:
        json_files = json_files[: args.n_tasks]

    all_results_by_model = {}

    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["Model", "Task_ID", "Ambiguity_Level", "Run_Index", "Utility_Score"]
        )

        for model_path, provider in zip(model_paths, providers):
            logging.info(f"\n==================================================")
            logging.info(f"Evaluating Model: {model_path} with Provider: {provider}")
            logging.info(f"==================================================")

            pipeline = setup_agentdojo_pipeline(model_path, provider=provider)
            all_results = {}

            try:
                for json_file in json_files:
                    res = evaluate_task_variations(
                        suite,
                        pipeline,
                        json_file,
                        base_output_dir,
                        args.n_runs,
                        csv_writer,
                        model_path,
                    )
                    if res:
                        task_id = json_file.stem.split(f"{args.agent}_")[-1]
                        all_results[task_id] = res

                all_results_by_model[model_path] = all_results
            except Exception as e:
                logging.error(f"Error evaluating model {model_path}: {e}")
                all_results_by_model[model_path] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY REPORT")
    print("=" * 50)

    overall_level_successes = {f"L{i}": 0 for i in range(1, 6)}
    overall_level_totals = {f"L{i}": 0 for i in range(1, 6)}

    agg_csv_path = base_output_dir / "aggregate_results.csv"
    with open(agg_csv_path, "w", newline="") as agg_csvfile:
        agg_writer = csv.writer(agg_csvfile)
        agg_writer.writerow(
            [
                "Scope",
                "Model",
                "Task_ID",
                "L1_Success",
                "L1_Total",
                "L2_Success",
                "L2_Total",
                "L3_Success",
                "L3_Total",
                "L4_Success",
                "L4_Total",
                "L5_Success",
                "L5_Total",
                "Total_Success",
                "Total_Runs",
                "Success_Rate",
            ]
        )

        for model_path, results in all_results_by_model.items():
            print(f"\n--- Model: {model_path} ---")

            if "error" in results:
                print(f"  [ERROR] Evaluation failed completely: {results['error']}")
                agg_writer.writerow(
                    ["Model Error", model_path, str(results["error"])] + [""] * 13
                )
                continue

            level_successes = {f"L{i}": 0 for i in range(1, 6)}
            level_totals = {f"L{i}": 0 for i in range(1, 6)}

            for task_id, task_results in results.items():
                print(f"\n  Task: {task_id}")
                task_successes = 0
                task_totals = 0
                t_succ = {f"L{i}": 0 for i in range(1, 6)}
                t_tot = {f"L{i}": 0 for i in range(1, 6)}

                for level, runs_dict in task_results.items():
                    level_passes = sum(1 for val in runs_dict.values() if val)
                    print(f"    {level}: PASSED {level_passes}/{args.n_runs}")
                    level_successes[level] += level_passes
                    level_totals[level] += args.n_runs
                    overall_level_successes[level] += level_passes
                    overall_level_totals[level] += args.n_runs
                    task_successes += level_passes
                    task_totals += args.n_runs
                    t_succ[level] += level_passes
                    t_tot[level] += args.n_runs

                if task_totals > 0:
                    task_rate = (task_successes / task_totals) * 100
                    print(
                        f"    Aggregate Task Success Rate: {task_successes}/{task_totals} ({task_rate:.1f}%)"
                    )
                    agg_writer.writerow(
                        [
                            "Per-Task",
                            model_path,
                            task_id,
                            t_succ.get("L1", 0),
                            t_tot.get("L1", 0),
                            t_succ.get("L2", 0),
                            t_tot.get("L2", 0),
                            t_succ.get("L3", 0),
                            t_tot.get("L3", 0),
                            t_succ.get("L4", 0),
                            t_tot.get("L4", 0),
                            t_succ.get("L5", 0),
                            t_tot.get("L5", 0),
                            task_successes,
                            task_totals,
                            f"{task_rate:.1f}%",
                        ]
                    )

            print(f"\n  Model Aggregate Success Rates:")
            tot_model_succ = 0
            tot_model_runs = 0
            for level in sorted(level_totals.keys()):
                if level_totals[level] > 0:
                    rate = (level_successes[level] / level_totals[level]) * 100
                    print(
                        f"    {level}: {level_successes[level]}/{level_totals[level]} ({rate:.1f}%)"
                    )
                    tot_model_succ += level_successes[level]
                    tot_model_runs += level_totals[level]

            if tot_model_runs > 0:
                mod_rate = (tot_model_succ / tot_model_runs) * 100
                agg_writer.writerow(
                    [
                        "Per-Model Aggregate",
                        model_path,
                        "ALL",
                        level_successes.get("L1", 0),
                        level_totals.get("L1", 0),
                        level_successes.get("L2", 0),
                        level_totals.get("L2", 0),
                        level_successes.get("L3", 0),
                        level_totals.get("L3", 0),
                        level_successes.get("L4", 0),
                        level_totals.get("L4", 0),
                        level_successes.get("L5", 0),
                        level_totals.get("L5", 0),
                        tot_model_succ,
                        tot_model_runs,
                        f"{mod_rate:.1f}%",
                    ]
                )

        print("\nOverall Aggregate Success Rates (All Models):")
        tot_all_succ = 0
        tot_all_runs = 0
        for level in sorted(overall_level_totals.keys()):
            if overall_level_totals[level] > 0:
                rate = (
                    overall_level_successes[level] / overall_level_totals[level]
                ) * 100
                print(
                    f"  {level}: {overall_level_successes[level]}/{overall_level_totals[level]} ({rate:.1f}%)"
                )
                tot_all_succ += overall_level_successes[level]
                tot_all_runs += overall_level_totals[level]

        if tot_all_runs > 0:
            agg_rate = (tot_all_succ / tot_all_runs) * 100
            agg_writer.writerow(
                [
                    "Overall Aggregate",
                    "ALL",
                    "ALL",
                    overall_level_successes.get("L1", 0),
                    overall_level_totals.get("L1", 0),
                    overall_level_successes.get("L2", 0),
                    overall_level_totals.get("L2", 0),
                    overall_level_successes.get("L3", 0),
                    overall_level_totals.get("L3", 0),
                    overall_level_successes.get("L4", 0),
                    overall_level_totals.get("L4", 0),
                    overall_level_successes.get("L5", 0),
                    overall_level_totals.get("L5", 0),
                    tot_all_succ,
                    tot_all_runs,
                    f"{agg_rate:.1f}%",
                ]
            )


if __name__ == "__main__":
    main()
