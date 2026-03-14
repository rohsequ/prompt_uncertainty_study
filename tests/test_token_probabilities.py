import argparse
import subprocess
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_best_cuda_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        free_memory = [int(x) for x in result.stdout.strip().split("\n")]
        best_gpu = free_memory.index(max(free_memory))
        return f"cuda:{best_gpu}"
    except subprocess.TimeoutExpired:
        print("[warn] nvidia-smi timed out. Falling back to cuda:0.")
        return "cuda:0"
    except Exception:
        return "cuda"


def run_forward_pass(model, tokenizer, input_ids_list: list[int]):
    """
    Run a single forward pass on the given token IDs.
    Returns:
        - generated_token_id: the argmax (greedy) next token
        - generated_token_str: its decoded string
        - all_probs: softmax probabilities tensor over the full vocabulary (for the last position)
    """
    input_tensor = torch.tensor([input_ids_list]).to(model.device)
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits[0, -1]  # logits at the last position

    probs = torch.softmax(logits, dim=-1)
    generated_token_id = torch.argmax(probs).item()
    generated_token_str = tokenizer.decode([generated_token_id])
    return generated_token_id, generated_token_str, probs


def print_top_tokens(tokenizer, probs, top_k=5):
    top_probs, top_ids = torch.topk(probs, top_k)
    print(f"  Top-{top_k} predictions from model:")
    for rank, (tid, tprob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), 1):
        print(f"    #{rank}: {repr(tokenizer.decode([tid])):<20} | ID: {tid:<6} | Prob: {tprob:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Step-by-step token probability calculator using incremental forward passes."
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the local HF model."
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The input prompt or context."
    )
    parser.add_argument(
        "--desired-output",
        type=str,
        required=True,
        help="The desired output string to evaluate token-by-token.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top model predictions to display at each step."
    )
    parser.add_argument(
        "--fast-thinking-skip",
        action="store_true",
        help="For reasoning models: append </think> directly to the prompt so the model "
             "is tricked into answer-delivery mode without generating any thinking tokens.",
    )
    parser.add_argument(
        "--low-thinking-skip",
        action="store_true",
        help="For reasoning models: generate the full thinking chain first, then use "
             "(prompt + thinking tokens) as context so probabilities are measured in "
             "genuine answer-delivery mode — no thinking tokens are skipped artificially.",
    )
    parser.add_argument(
        "--low-thinking-max-tokens",
        type=int,
        default=2048,
        help="Max tokens for the thinking-generation pass in --low-thinking-skip mode (default: 2048).",
    )

    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading model from {args.model_path} with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
    )
    model.eval()

    prompt_text = args.prompt
    target_text = args.desired_output

    # Apply chat template if the tokenizer supports it (instruct models)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"\n[Chat-formatted Prompt]\n  {repr(formatted_prompt)}")
    else:
        formatted_prompt = prompt_text
        print("\n[No chat template found — using raw prompt]")

    # ── Fast thinking skip: inject </think> token so model never enters thinking mode ──
    if args.fast_thinking_skip:
        formatted_prompt = formatted_prompt + "</think>\n\n"
        print("[Fast thinking skip] Appended '</think>' directly — model skips thinking entirely.")
        print(f"[Effective prompt ends with]\n  {repr(formatted_prompt[-60:])}")

    # ── Low thinking skip: generate real thinking chain, then use it as context ────────
    elif args.low_thinking_skip:
        print("[Low thinking skip] Generating thinking chain (this may take a moment)...")
        think_input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        think_tensor = torch.tensor([think_input_ids]).to(model.device)
        with torch.no_grad():
            think_output = model.generate(
                think_tensor,
                max_new_tokens=args.low_thinking_max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = think_output[0][len(think_input_ids):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        print(f"[Generated output (first 300 chars)]\n  {repr(generated_text[:300])}")

        # Isolate only the thinking portion (up to and including </think>)
        think_end_marker = "</think>"
        think_end_pos = generated_text.find(think_end_marker)
        if think_end_pos == -1:
            print("[Warning] '</think>' not found in generated output — using full generation as context.")
            thinking_text = generated_text
        else:
            thinking_text = generated_text[: think_end_pos + len(think_end_marker)] + "\n\n"

        # NOTE: thinking_text is NOT appended to formatted_prompt here.
        # It will be added to prompt_ids AFTER the full model generation display below.
        print(f"[Thinking chain captured ({len(thinking_text)} chars)]")

    print(f"\n[Original Prompt]\n  {repr(prompt_text)}")
    print(f"[Desired Output]\n  {repr(target_text)}")

    # Tokenize the formatted prompt and the full string
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(formatted_prompt + target_text, add_special_tokens=False)

    # Find how many tokens the prompt occupies in the concatenated string
    match_len = sum(1 for p, f in zip(prompt_ids, full_ids) if p == f)
    target_ids = full_ids[match_len:]

    print(f"\nPrompt tokens ({len(prompt_ids)}): {prompt_ids}")
    print(f"Target tokens ({len(target_ids)}): {target_ids}")
    print(f"Target token strings: {[tokenizer.decode([t]) for t in target_ids]}")

    # ─── Full generation before Step 0 ───────────────────────────────────────
    print("\n" + "="*70)
    print("FULL MODEL GENERATION (before step-by-step analysis)")
    print("="*70)
    gen_input_tensor = torch.tensor([prompt_ids]).to(model.device)
    with torch.no_grad():
        gen_output_ids = model.generate(
            gen_input_tensor,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = gen_output_ids[0][len(prompt_ids):]
    full_output_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    if args.low_thinking_skip:
        # Show the thinking chain the model produced, then the answer it would give
        print(f"[Thinking chain]\n  {repr(thinking_text)}")
        print(f"[Answer from original prompt]\n  {repr(full_output_text)}")
        # Now extend prompt_ids with thinking tokens so step-by-step uses proper context
        prompt_ids = tokenizer.encode(formatted_prompt + thinking_text, add_special_tokens=False)
        full_ids = tokenizer.encode(formatted_prompt + thinking_text + target_text, add_special_tokens=False)
        match_len = sum(1 for p, f in zip(prompt_ids, full_ids) if p == f)
        target_ids = full_ids[match_len:]
        print(f"\n[Updated prompt tokens with thinking ({len(prompt_ids)} total)]")
        print(f"Target tokens ({len(target_ids)}): {[tokenizer.decode([t]) for t in target_ids]}")
    else:
        print(f"  {repr(full_output_text)}")

    # # ─── Step 0: Base forward pass on the raw prompt ─────────────────────────
    # print("\n" + "="*70)
    # print("STEP 0: Forward pass on raw prompt")
    # print("="*70)

    # gen_id, gen_str, base_probs = run_forward_pass(model, tokenizer, prompt_ids)
    # print(f"  Model greedy next token: {repr(gen_str)} (ID: {gen_id})")
    # print_top_tokens(tokenizer, base_probs, args.top_k)

    # if target_ids:
    #     first_target_id = target_ids[0]
    #     first_target_prob = base_probs[first_target_id].item()
    #     first_target_logprob = math.log(first_target_prob) if first_target_prob > 0 else float("-inf")
    #     first_target_str = tokenizer.decode([first_target_id])
    #     print(f"\n  Probability of desired token {repr(first_target_str)} (ID: {first_target_id}): "
    #           f"Prob={first_target_prob:.6f} | LogProb={first_target_logprob:.6f}")

    # ─── Loop: extend with each desired token and re-run ─────────────────────
    total_log_prob = 0.0
    token_results = []

    for step, target_id in enumerate(target_ids):
        # Input at this step: prompt + all desired tokens up to (but not including) current
        current_input_ids = prompt_ids + target_ids[:step]
        next_target_id = target_id  # the desired token we want at this position
        
        print("\n" + "="*70)
        print(f"STEP {step + 1}: Evaluating desired token {step + 1}/{len(target_ids)}")
        print(f"  Input: prompt + {[tokenizer.decode([t]) for t in target_ids[:step]]}")
        print("="*70)

        gen_id, gen_str, step_probs = run_forward_pass(model, tokenizer, current_input_ids)

        print(f"  Model greedy next token: {repr(gen_str)} (ID: {gen_id})")
        print_top_tokens(tokenizer, step_probs, args.top_k)

        desired_prob = step_probs[next_target_id].item()
        desired_logprob = math.log(desired_prob) if desired_prob > 0 else float("-inf")
        desired_str = tokenizer.decode([next_target_id])

        total_log_prob += desired_logprob
        token_results.append((desired_str, next_target_id, desired_prob, desired_logprob))

        print(f"\n  ➜ Desired token {repr(desired_str)} (ID: {next_target_id}): "
              f"Prob={desired_prob:.6f} | LogProb={desired_logprob:.6f}")

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Token':<20} | {'ID':<6} | {'Probability':>12} | {'LogProb':>12}")
    print("-"*60)
    for token_str, token_id, prob, logprob in token_results:
        print(f"{repr(token_str):<20} | {token_id:<6} | {prob:>12.6f} | {logprob:>12.6f}")
    print("-"*60)
    overall_prob = math.exp(total_log_prob) if total_log_prob > float("-inf") else 0.0
    print(f"\nOverall joint probability of desired output: {overall_prob:.6e}")
    print(f"Total log probability: {total_log_prob:.6f}")


if __name__ == "__main__":
    main()
