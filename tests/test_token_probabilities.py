import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import subprocess
import math


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
        print(
            "[warn] nvidia-smi timed out after 5s due to locked PCIe bus. Dropping to cuda:0 fallback."
        )
        return "cuda:0"
    except Exception:
        return "cuda"


def main():
    parser = argparse.ArgumentParser(
        description="Calculate token probabilities for a desired output string."
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
        help="The exact desired output string to calculate probabilities for.",
    )

    args = parser.parse_args()

    device = get_best_cuda_device()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading model from {args.model_path} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Apply chat template if available, or just use raw strings
    # For a pure probability check, we should construct the full expected string
    # Assuming the prompt is the exact string to be fed to the model (including any chat formatting if necessary)

    prompt_text = args.prompt
    target_text = args.desired_output

    full_text = prompt_text + target_text

    print("\n[Input Text]\n", prompt_text)
    print("\n[Desired Output]\n", target_text)

    # Tokenize prompt and full text
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)

    # Verify how the tokenizer handled the concatenation
    # Sometimes tokenizing them separately differs from together due to space prefixes
    prompt_len = len(prompt_ids)
    target_ids = full_ids[prompt_len:]

    print(f"\nPrompt length: {prompt_len} tokens")
    print(f"Target length: {len(target_ids)} tokens")

    # Run forward pass
    input_tensor = torch.tensor([full_ids]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits[0]  # shape: (seq_len, vocab_size)

    print("\n--- Token Probabilities ---")
    total_log_prob = 0.0

    # logits[i] corresponds to the prediction for token i+1
    for i in range(prompt_len - 1, len(full_ids) - 1):
        target_token_id = full_ids[i + 1]

        # Get logits for the prediction step
        step_logits = logits[i]

        # Calculate probabilities
        probs = torch.softmax(step_logits, dim=-1)

        # Get the probability of the actual target token
        token_prob = probs[target_token_id].item()

        # Log probability
        log_prob = torch.log(probs[target_token_id]).item()
        total_log_prob += log_prob

        # Decode the token for display
        token_str = tokenizer.decode([target_token_id])

        print(
            f"Token: {repr(token_str):<15} | ID: {target_token_id:<6} | Probability: {token_prob:.6f} | LogProb: {log_prob:.6f}"
        )

    # Calculate overall sequence probability (assuming independent step generation given context)
    overall_prob = math.exp(total_log_prob)
    print(
        f"\nOverall Probability of generating the exact desired output: {overall_prob:.6e}"
    )
    print(f"Total Log Probability: {total_log_prob:.6f}")


if __name__ == "__main__":
    main()
