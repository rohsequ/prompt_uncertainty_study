import argparse
import json
import logging
from pathlib import Path
from copy import deepcopy
import datetime
import csv
import os
import sys

from agentdojo.benchmark import (
    run_task_with_injection_tasks,
    run_task_without_injection_tasks,
)
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.attacks.attack_registry import load_attack

from utils.pipeline_utils import setup_agentdojo_pipeline
from utils.custom_logger import FullOutputLogger

logging.basicConfig(level=logging.INFO)

BENCHMARK_VERSION = "v1.2.2"


def evaluate_task_variations(
    suite,
    pipeline,
    attack,
    task_json_path: Path,
    output_dir: Path,
    n_runs: int,
    csv_writer,
    model_name: str,
    force_rerun: bool = True,
):
    """
    Evaluates all L1-L5 prompt variations defined in a task JSON file n_runs times against prompt injections.
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
                task_copy_no_inj = deepcopy(original_task)
                task_copy_inj = deepcopy(original_task)

                # Override properties for this ambiguity level run
                task_copy_no_inj.ID = f"{task_id}_{level}_run{run_idx}_no_inj"
                task_copy_no_inj.PROMPT = new_prompt

                task_copy_inj.ID = f"{task_id}_{level}_run{run_idx}_inj"
                task_copy_inj.PROMPT = new_prompt

                # Utility without attack
                try:
                    utility_no_attack, _ = run_task_without_injection_tasks(
                        suite=suite,
                        agent_pipeline=pipeline,
                        task=task_copy_no_inj,
                        logdir=output_dir,
                        force_rerun=force_rerun,
                        benchmark_version=BENCHMARK_VERSION,
                    )
                except Exception as e:
                    logging.error(
                        f"Error executing task without attack {task_copy_no_inj.ID}: {e}"
                    )
                    utility_no_attack = False

                # Run the agentdojo evaluation loop with injections
                try:
                    utility_res, security_res = run_task_with_injection_tasks(
                        suite=suite,
                        agent_pipeline=pipeline,
                        user_task=task_copy_inj,
                        attack=attack,
                        logdir=output_dir,
                        force_rerun=force_rerun,
                        benchmark_version=BENCHMARK_VERSION,
                    )
                except Exception as e:
                    logging.error(
                        f"Error executing injection task {task_copy_inj.ID}: {e}"
                    )
                    utility_res = {
                        (task_copy_inj.ID, inj_id): False
                        for inj_id in suite.injection_tasks.keys()
                    }
                    security_res = {
                        (task_copy_inj.ID, inj_id): False
                        for inj_id in suite.injection_tasks.keys()
                    }

                results[level][run_idx] = {}

                for key, util_atk in utility_res.items():
                    _, inj_task_id = key

                    # AgentDojo sets security to True if the injection succeeds, BUT only if
                    # the malicious criteria are met. Thus, an attack is only truly 'successful'
                    # if the injection criteria (security=True) is met AND the model
                    # actually completed the task (util_atk=True). If util_atk is False (e.g. model crashed/refused),
                    # the attack fundamentally failed.

                    atk_succ = security_res.get(key, False)
                    if not util_atk:
                        atk_succ = False

                    # Utility without attack should only be True if the base run (utility_no_attack) was True
                    # (This is already inherently handled by the variable utility_no_attack)

                    results[level][run_idx][inj_task_id] = {
                        "utility_without_attack": utility_no_attack,
                        "utility_under_attack": util_atk,
                        "attack_success": atk_succ,
                    }

                    print(
                        f"[{level} | Run {run_idx} | Inj {inj_task_id}] UtilNoAtk: {utility_no_attack} | UtilAtk: {util_atk} | AtkSucc: {atk_succ}"
                    )

                    # Write to CSV tracking artifact
                    csv_writer.writerow(
                        [
                            model_name,
                            task_id,
                            level,
                            run_idx,
                            inj_task_id,
                            int(utility_no_attack),
                            int(util_atk),
                            int(atk_succ),
                        ]
                    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Ambiguity Prompts with Injections on AgentDojo"
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
        default=Path("runs_eval_injection"),
        help="Output directory for logs",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Optional path to an existing results directory to resume from. Will skip already completed tasks and recreate the global CSV files.",
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
        default="gpt-5-mini",
        help="Model string to use for prompt generation. Default: gpt-5-mini",
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
    parser.add_argument(
        "--attack",
        type=str,
        default="direct",
        help="Attack to load (e.g. direct, naive, dos). Default: direct",
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

    if args.resume_dir:
        base_output_dir = args.resume_dir
        force_rerun = False
        logging.info(f"Resuming evaluation from existing directory: {base_output_dir}")
        if not base_output_dir.exists():
            logging.error(
                f"Provided resume directory does not exist: {base_output_dir}"
            )
            sys.exit(1)
    else:
        # Base log dir: results_TIME
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = args.output_dir / f"results_{timestamp}"
        base_output_dir.mkdir(parents=True, exist_ok=True)
        force_rerun = True

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
            [
                "Model",
                "Task_ID",
                "Ambiguity_Level",
                "Run_Index",
                "Injection_Task_ID",
                "Utility_Without_Attack",
                "Utility_Under_Attack",
                "Attack_Success",
            ]
        )

        for model_path, provider in zip(model_paths, providers):
            logging.info(f"\n==================================================")
            logging.info(
                f"Evaluating Model: {model_path} with Provider: {provider} against Attack: {args.attack}"
            )
            logging.info(f"==================================================")

            pipeline = setup_agentdojo_pipeline(model_path, provider=provider)
            attack = load_attack(args.attack, suite, pipeline)

            all_results = {}

            try:
                for json_file in json_files:
                    res = evaluate_task_variations(
                        suite,
                        pipeline,
                        attack,
                        json_file,
                        base_output_dir,
                        args.n_runs,
                        csv_writer,
                        model_path,
                        force_rerun,
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

    ovr_succ_no_atk = {f"L{i}": 0 for i in range(1, 6)}
    ovr_succ_atk = {f"L{i}": 0 for i in range(1, 6)}
    ovr_succ_atk_succ = {f"L{i}": 0 for i in range(1, 6)}
    ovr_totals = {f"L{i}": 0 for i in range(1, 6)}

    agg_csv_path = base_output_dir / "aggregate_results.csv"
    with open(agg_csv_path, "w", newline="") as agg_csvfile:
        agg_writer = csv.writer(agg_csvfile)

        headers = ["Scope", "Model", "Task_ID"]
        for i in range(1, 6):
            headers += [
                f"L{i}_UtilNoAtk",
                f"L{i}_UtilAtk",
                f"L{i}_AtkSucc",
                f"L{i}_Total",
            ]
        headers += ["Total_UtilNoAtk", "Total_UtilAtk", "Total_AtkSucc", "Total_Runs"]

        agg_writer.writerow(headers)

        for model_path, results in all_results_by_model.items():
            print(f"\n--- Model: {model_path} ---")

            if "error" in results:
                print(f"  [ERROR] Evaluation failed completely: {results['error']}")
                agg_writer.writerow(
                    ["Model Error", model_path, str(results["error"])] + [""] * 24
                )
                continue

            mdl_succ_no_atk = {f"L{i}": 0 for i in range(1, 6)}
            mdl_succ_atk = {f"L{i}": 0 for i in range(1, 6)}
            mdl_succ_atk_succ = {f"L{i}": 0 for i in range(1, 6)}
            mdl_totals = {f"L{i}": 0 for i in range(1, 6)}

            for task_id, task_results in results.items():
                print(f"\n  Task: {task_id}")
                tsk_succ_no_atk = 0
                tsk_succ_atk = 0
                tsk_succ_atk_succ = 0
                tsk_totals = 0

                t_no = {f"L{i}": 0 for i in range(1, 6)}
                t_at = {f"L{i}": 0 for i in range(1, 6)}
                t_su = {f"L{i}": 0 for i in range(1, 6)}
                t_to = {f"L{i}": 0 for i in range(1, 6)}

                for level, runs_dict in task_results.items():
                    l_no = sum(
                        1
                        for run_dict in runs_dict.values()
                        for val in run_dict.values()
                        if val["utility_without_attack"]
                    )
                    l_at = sum(
                        1
                        for run_dict in runs_dict.values()
                        for val in run_dict.values()
                        if val["utility_under_attack"]
                    )
                    l_su = sum(
                        1
                        for run_dict in runs_dict.values()
                        for val in run_dict.values()
                        if val["attack_success"]
                    )
                    l_runs = sum(len(run_dict) for run_dict in runs_dict.values())

                    print(
                        f"    {level}: UtilNoAtk {l_no}/{l_runs} | UtilAtk {l_at}/{l_runs} | AtkSucc {l_su}/{l_runs}"
                    )

                    mdl_succ_no_atk[level] += l_no
                    mdl_succ_atk[level] += l_at
                    mdl_succ_atk_succ[level] += l_su
                    mdl_totals[level] += l_runs

                    ovr_succ_no_atk[level] += l_no
                    ovr_succ_atk[level] += l_at
                    ovr_succ_atk_succ[level] += l_su
                    ovr_totals[level] += l_runs

                    tsk_succ_no_atk += l_no
                    tsk_succ_atk += l_at
                    tsk_succ_atk_succ += l_su
                    tsk_totals += l_runs

                    t_no[level] += l_no
                    t_at[level] += l_at
                    t_su[level] += l_su
                    t_to[level] += l_runs

                if tsk_totals > 0:
                    row = ["Per-Task", model_path, task_id]
                    for i in range(1, 6):
                        row += [
                            t_no[f"L{i}"],
                            t_at[f"L{i}"],
                            t_su[f"L{i}"],
                            t_to[f"L{i}"],
                        ]
                    row += [
                        tsk_succ_no_atk,
                        tsk_succ_atk,
                        tsk_succ_atk_succ,
                        tsk_totals,
                    ]
                    agg_writer.writerow(row)

            print(f"\n  Model Aggregate Metrics:")
            mod_no = 0
            mod_at = 0
            mod_su = 0
            mod_to = 0
            for level in sorted(mdl_totals.keys()):
                if mdl_totals[level] > 0:
                    print(
                        f"    {level}: UtilNoAtk {mdl_succ_no_atk[level]}/{mdl_totals[level]} | UtilAtk {mdl_succ_atk[level]}/{mdl_totals[level]} | AtkSucc {mdl_succ_atk_succ[level]}/{mdl_totals[level]}"
                    )
                    mod_no += mdl_succ_no_atk[level]
                    mod_at += mdl_succ_atk[level]
                    mod_su += mdl_succ_atk_succ[level]
                    mod_to += mdl_totals[level]

            if mod_to > 0:
                row = ["Per-Model Aggregate", model_path, "ALL"]
                for i in range(1, 6):
                    row += [
                        mdl_succ_no_atk[f"L{i}"],
                        mdl_succ_atk[f"L{i}"],
                        mdl_succ_atk_succ[f"L{i}"],
                        mdl_totals[f"L{i}"],
                    ]
                row += [mod_no, mod_at, mod_su, mod_to]
                agg_writer.writerow(row)

        print("\nOverall Aggregate Metrics (All Models):")
        all_no = 0
        all_at = 0
        all_su = 0
        all_to = 0
        for level in sorted(ovr_totals.keys()):
            if ovr_totals[level] > 0:
                print(
                    f"  {level}: UtilNoAtk {ovr_succ_no_atk[level]}/{ovr_totals[level]} | UtilAtk {ovr_succ_atk[level]}/{ovr_totals[level]} | AtkSucc {ovr_succ_atk_succ[level]}/{ovr_totals[level]}"
                )
                all_no += ovr_succ_no_atk[level]
                all_at += ovr_succ_atk[level]
                all_su += ovr_succ_atk_succ[level]
                all_to += ovr_totals[level]

        if all_to > 0:
            row = ["Overall Aggregate", "ALL", "ALL"]
            for i in range(1, 6):
                row += [
                    ovr_succ_no_atk[f"L{i}"],
                    ovr_succ_atk[f"L{i}"],
                    ovr_succ_atk_succ[f"L{i}"],
                    ovr_totals[f"L{i}"],
                ]
            row += [all_no, all_at, all_su, all_to]
            agg_writer.writerow(row)


if __name__ == "__main__":
    main()
