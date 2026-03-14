import argparse
import math
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(
        description="Calculate token probabilities for a desired output string using vLLM."
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

    # Load model with vLLM
    # tensor_parallel_size=2 because we have 2 GPUs available, scaling to the 27B model sizes
    print(f"Loading vLLM model from {args.model_path} with tensor_parallel_size=2...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
    )

    tokenizer = llm.get_tokenizer()

    prompt_text = args.prompt
    target_text = args.desired_output

    full_text = prompt_text + target_text

    print("\n[Input Text]\n", prompt_text)
    print("\n[Desired Output]\n", target_text)

    # Tokenizer processing
    prompt_ids = tokenizer.encode(prompt_text)
    full_ids = tokenizer.encode(full_text)

    # Find matching prefix length exactly like before
    match_len = 0
    for p_id, f_id in zip(prompt_ids, full_ids):
        if p_id == f_id:
            match_len += 1
        else:
            break
            
    prompt_len = match_len
    target_ids = full_ids[prompt_len:]

    print(f"\nPrompt length (matching prefix): {prompt_len} tokens")
    print(f"Target length: {len(target_ids)} tokens")
    print("Target tokens:", [tokenizer.decode([t]) for t in target_ids])

    # We use vLLM to compute prompt logprobs
    # By passing the full_text as prompt and setting prompt_logprobs > 0
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1, # We only care about the prompt evaluation
        prompt_logprobs=1, # Request logprobs for the prompt tokens
    )

    print("\n--- Running Forward Pass (vLLM) ---")
    # Pass the full_ids as the prompt to avoid re-tokenization differences
    outputs = llm.generate([{"prompt_token_ids": full_ids}], sampling_params=sampling_params)
    output = outputs[0]

    # prompt_logprobs is a list of dictionaries mapping token_id to Logprob
    # The first element corresponds to the first token, which doesn't have a logprob
    prompt_logprobs = output.prompt_logprobs

    print("\n--- Token Probabilities ---")
    total_log_prob = 0.0
    
    # We evaluate from prompt_len to end
    for i in range(prompt_len, len(full_ids)):
        target_token_id = full_ids[i]
        
        # logprobs for predicting full_ids[i] based on full_ids[:i] is stored at prompt_logprobs[i]
        step_logprobs = prompt_logprobs[i]
        
        if step_logprobs and target_token_id in step_logprobs:
            log_prob = step_logprobs[target_token_id].logprob
            token_prob = math.exp(log_prob)
        else:
            # Fallback if the token is completely missing (unlikely in vLLM since it includes the prompt token by default)
            log_prob = float("-inf")
            token_prob = 0.0

        total_log_prob += log_prob
        
        # Handle tokenizer.decode requiring lists or single integers properly
        token_str = tokenizer.decode([target_token_id])
        
        if step_logprobs:
            # Find top predicted via dict values
            top_token_id = max(step_logprobs.keys(), key=lambda k: step_logprobs[k].logprob)
            top_log_prob = step_logprobs[top_token_id].logprob
            top_token_prob = math.exp(top_log_prob)
            top_token_str = tokenizer.decode([top_token_id])
        else:
            top_token_id = target_token_id
            top_token_prob = 0.0
            top_token_str = "N/A"

        print(f"\nStep {i - prompt_len + 1}:")
        print(f"  Desired: {repr(token_str):<15} | ID: {target_token_id:<6} | Prob: {token_prob:.6f} | LogProb: {log_prob:.6f}")
        print(f"  TopPred: {repr(top_token_str):<15} | ID: {top_token_id:<6} | Prob: {top_token_prob:.6f}")

    # Calculate overall sequence probability
    overall_prob = math.exp(total_log_prob) if total_log_prob > float("-inf") else 0.0
    print(f"\nOverall Probability of generating the exact desired output: {overall_prob:.6e}")
    print(f"Total Log Probability: {total_log_prob:.6f}")

if __name__ == "__main__":
    main()
