#!/usr/bin/env python3
"""
Script to send prompts to vLLM server
"""
import json
import requests
from typing import List, Dict, Optional
import time
from tqdm import tqdm
# [NOTE, hyunnnchoi, 2025.11.12] Added tokenizer for output slicing
from transformers import AutoTokenizer
# [NOTE, hyunnnchoi, 2025.11.12] Added module for parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed

# [NOTE, hyunnnchoi, 2025.11.12] vLLM server configuration
VLLM_SERVER_URL = "http://localhost:8000/v1/completions"  # vLLM server URL
# [NOTE, hyunnnchoi, 2025.11.12] Changed model name to gpt-oss-20b
MODEL_NAME = "gpt-oss-20b"  # Model name to use


def load_prompts(json_file_path: str) -> List[str]:
    """
    Loads a list of prompts from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        List of prompt strings
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('prompts', [])


# [NOTE, hyunnnchoi, 2025.11.12] Added function to slice output cumulatively by 50 tokens
def slice_output_by_tokens(output_text: str, tokenizer, chunk_size: int = 50) -> List[Dict]:
    """
    Slices the output text cumulatively by 50 tokens and returns information for each chunk.
    
    Args:
        output_text: Full output text
        tokenizer: Tokenizer to use
        chunk_size: Chunk size (default: 50)
        
    Returns:
        List of dictionaries containing information for each chunk
        [
            {"output_text": "0~50 tokens", "num_tokens": 50, "remaining_tokens": 100},
            {"output_text": "0~100 tokens", "num_tokens": 100, "remaining_tokens": 50},
            ...
        ]
    """
    # Tokenize output text
    tokens = tokenizer.encode(output_text, add_special_tokens=False)
    total_tokens = len(tokens)
    
    chunks = []
    current_pos = 0
    
    # Slice cumulatively by 50 tokens
    while current_pos < total_tokens:
        next_pos = min(current_pos + chunk_size, total_tokens)
        
        # Decode tokens from 0 to next_pos
        chunk_tokens = tokens[:next_pos]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        chunks.append({
            "output_text": chunk_text,
            "num_tokens": next_pos,
            "remaining_tokens": total_tokens - next_pos
        })
        
        current_pos = next_pos
    
    # If the last chunk is not exactly total_tokens, add the whole text
    if not chunks or chunks[-1]["num_tokens"] < total_tokens:
        full_text = tokenizer.decode(tokens, skip_special_tokens=True)
        chunks.append({
            "output_text": full_text,
            "num_tokens": total_tokens,
            "remaining_tokens": 0
        })
    
    return chunks


def send_to_vllm(
    prompt: str,
    server_url: str = VLLM_SERVER_URL,
    model_name: str = MODEL_NAME,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout: int = 300
) -> Optional[Dict]:
    """
    Sends a single prompt to the vLLM server.
    
    Args:
        prompt: Prompt to send
        server_url: vLLM server URL
        model_name: Model name
        max_tokens: Maximum tokens to generate (None for no limit)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        timeout: Request timeout (seconds)
        
    Returns:
        API response or None (if error occurs)
    """
    # [NOTE, hyunnnchoi, 2025.11.12] Added option to remove max_tokens limit
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    # Add only if max_tokens is specified
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    
    try:
        response = requests.post(
            server_url, 
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        return None


def process_all_prompts(
    json_file_path: str,
    output_file_path: str,
    server_url: str = VLLM_SERVER_URL,
    model_name: str = MODEL_NAME,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    batch_delay: float = 0.0,
    tokenizer_path: str = None,
    batch_size: int = 16
):
    """
    Processes all prompts and saves the results.
    
    Args:
        json_file_path: Input JSON file path
        output_file_path: Output JSONL file path
        server_url: vLLM server URL
        model_name: Model name
        max_tokens: Maximum tokens to generate (None for no limit)
        temperature: Sampling temperature
        batch_delay: Wait time between requests (seconds)
        tokenizer_path: Tokenizer path (if None, use model_name)
        batch_size: Number of requests to process concurrently
    """
    # [NOTE, hyunnnchoi, 2025.11.12] Load tokenizer
    # [NOTE, hyunnnchoi, 2025.11.16] Modified to use model_name when tokenizer_path is missing
    print(f"🔧 Loading Tokenizer...")
    if tokenizer_path is None:
        tokenizer_path = model_name  # Use model_name as tokenizer path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"✅ Tokenizer loaded: {tokenizer_path}\n")
    except Exception as e:
        print(f"⚠️ Tokenizer load failed, using default tokenizer: {e}\n")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # fallback
    
    print(f"📂 Loading file: {json_file_path}")
    prompts = load_prompts(json_file_path)
    print(f"✅ Loaded {len(prompts)} prompts.\n")
    
    # [NOTE, hyunnnchoi, 2025.11.12] Open file first for real-time saving
    training_output_path = output_file_path.replace('.json', '_training.jsonl')
    print(f"💾 Creating training data file: {training_output_path}")
    training_file = open(training_output_path, 'w', encoding='utf-8')
    
    results = []
    success_count = 0
    fail_count = 0
    total_training_samples = 0
    
    print(f"🚀 Starting request transmission to vLLM server...")
    print(f"   Server URL: {server_url}")
    print(f"   Model: {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max tokens: {'No limit' if max_tokens is None else max_tokens}\n")
    
    # [NOTE, hyunnnchoi, 2025.11.12] Helper function for parallel processing
    def process_single_prompt(idx_prompt_tuple):
        idx, prompt = idx_prompt_tuple
        response = send_to_vllm(
            prompt=prompt,
            server_url=server_url,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return idx, prompt, response
    
    # [NOTE, hyunnnchoi, 2025.11.12] Parallel processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Submit all prompts with index
        futures = {
            executor.submit(process_single_prompt, (idx, prompt)): idx
            for idx, prompt in enumerate(prompts)
        }
        
        # Process completed tasks
        for future in tqdm(as_completed(futures), total=len(prompts), desc="Processing"):
            try:
                idx, prompt, response = future.result()
                
                # [NOTE, hyunnnchoi, 2025.11.12] Save input/output in real-time
                if response:
                    # Extract generated text
                    output_text = ""
                    if "choices" in response and len(response["choices"]) > 0:
                        output_text = response["choices"][0].get("text", "")
                    
                    # Slice output cumulatively by 50 tokens
                    if output_text.strip():
                        chunks = slice_output_by_tokens(output_text, tokenizer, chunk_size=50)
                        
                        # Write each chunk directly to file (real-time saving)
                        for chunk in chunks:
                            training_entry = {
                                "input_prompt": prompt,
                                "output_prompt": chunk["output_text"],
                                "number_of_output_tokens": chunk["num_tokens"],
                                "remaining_tokens": chunk["remaining_tokens"]
                            }
                            training_file.write(json.dumps(training_entry, ensure_ascii=False) + '\n')
                            training_file.flush()  # Flush buffer to disk immediately
                            total_training_samples += 1
                    
                    result_entry = {
                        "index": idx,
                        "input_prompt": prompt,
                        "output_text": output_text,
                        "full_response": response,
                        "status": "success"
                    }
                    success_count += 1
                else:
                    result_entry = {
                        "index": idx,
                        "input_prompt": prompt,
                        "output_text": "",
                        "error": "Request failed",
                        "status": "failed"
                    }
                    fail_count += 1
                
                results.append(result_entry)
                
            except Exception as e:
                print(f"\n❌ Exception occurred (idx={futures[future]}): {str(e)}")
                result_entry = {
                    "index": futures[future],
                    "input_prompt": prompts[futures[future]],
                    "output_text": "",
                    "error": str(e),
                    "status": "failed"
                }
                results.append(result_entry)
                fail_count += 1
    
    # [NOTE, hyunnnchoi, 2025.11.12] Real-time saving completed, close file
    training_file.close()
    print(f"\n✅ Real-time saving of training data completed: {training_output_path}")
    
    # Save original results as JSON (for debugging)
    print(f"💾 Saving original results: {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(prompts),
            "success": success_count,
            "failed": fail_count,
            "total_training_samples": total_training_samples,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Completed!")
    print(f"   Total prompts: {len(prompts)}")
    print(f"   Success: {success_count}, Failed: {fail_count}")
    print(f"   Training data samples: {total_training_samples}")
    print(f"   Training data file: {training_output_path}")
    print(f"   Original results file: {output_file_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Send prompts to vLLM server")
    parser.add_argument(
        "--input",
        type=str,
        default="/data/processed_dataset.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/vllm_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=VLLM_SERVER_URL,
        help="vLLM server URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model name"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: None, no limit)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Wait time between requests (seconds)"
    )
    # [NOTE, hyunnnchoi, 2025.11.12] Added tokenizer path argument
    # [NOTE, hyunnnchoi, 2025.11.16] Modified help message
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path (default: same as --model)"
    )
    # [NOTE, hyunnnchoi, 2025.11.12] Added batch size argument
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of requests to process concurrently (default: 16)"
    )
    
    args = parser.parse_args()
    
    process_all_prompts(
        json_file_path=args.input,
        output_file_path=args.output,
        server_url=args.server_url,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_delay=args.delay,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size
    )

