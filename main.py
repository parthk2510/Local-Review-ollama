import os
import json
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:270m"

IGNORE_DIRS = {
    "venv", ".venv", "env",
    "node_modules",
    "__pycache__",
    ".git",
    "logs",
    "instance",
    "build",
    "dist",
    ".next",
    ".cache"
}

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx",
    ".json", ".yaml", ".yml"
}

MAX_FILE_SIZE = 15000
DEFAULT_MAX_TOKENS_PER_BATCH = 6000
OUTPUT_TOKEN_RESERVE = 1024
OLLAMA_NUM_PARALLEL = 2

def should_ignore(path):
    parts = set(Path(path).parts)
    return bool(parts & IGNORE_DIRS)

def collect_files(root_dir, target_file=None):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            if target_file and file != target_file:
                continue
            if not any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                continue
            full_path = os.path.join(root, file)
            if should_ignore(full_path):
                continue
            file_paths.append(full_path)
    return file_paths

def read_file_content(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content) > MAX_FILE_SIZE:
                content = content[:MAX_FILE_SIZE] + "\nTRUNCATED"
            return (path, content)
    except Exception as e:
        return (path, f"ERROR: {str(e)}")

def read_files_parallel(file_paths, max_workers=8):
    data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(read_file_content, p): p for p in file_paths}
        with tqdm(total=len(file_paths), desc="Reading files") as pbar:
            for future in as_completed(future_to_path):
                path, content = future.result()
                data[path] = content
                pbar.update(1)
    return data

def get_structure(root_dir):
    structure = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        level = root.replace(root_dir, "").count(os.sep)
        indent = " " * 2 * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 2 * (level + 1)
        for file in files:
            if any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                structure.append(f"{sub_indent}{file}")
    return "\n".join(structure)

def estimate_tokens(text):
    return len(text) // 4

def build_prompt_strict(structure, batch_data):
    prompt = f"""
SYSTEM ROLE:
You are a deterministic static code analyzer. You do NOT guess. You ONLY report issues that are directly supported by the provided code.

HARD RULES:
- Do NOT assume missing context.
- Do NOT infer behavior not visible in code.
- If unsure, write: "INSUFFICIENT EVIDENCE".
- Every issue MUST include:
  1. File path
  2. Exact code reference (line or snippet)
  3. Why it is an issue (technical reasoning)
- Do NOT produce generic advice.
- Do NOT repeat the same issue.
- Be concise and factual.

ANALYSIS METHOD (FOLLOW STRICTLY):
1. Read each file independently
2. Then check cross-file dependencies
3. Only report issues you can prove from code

DIRECTORY STRUCTURE:
{structure}

OUTPUT FORMAT (STRICT JSON, NO EXTRA TEXT):

{{
  "critical_bugs": [
    {{
      "file": "",
      "evidence": "",
      "reason": ""
    }}
  ],
  "dependency_issues": [],
  "security_risks": [],
  "performance_problems": [],
  "code_quality": [],
  "architecture_issues": [],
  "refactor_plan": [],
  "priority_fix_order": []
}}

FILES:
"""
    for path, code in batch_data.items():
        prompt += f"\nFILE: {path}\n```\n{code}\n```"
    prompt += """

FINAL INSTRUCTION:
- Output ONLY valid JSON.
- No explanations outside JSON.
- If no issues found in a section, return empty list [].
"""
    return prompt

def query_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=120
    )
    if response.status_code != 200:
        raise Exception(f"Ollama API error {response.status_code}: {response.text}")
    result = response.json()
    return result.get("response", "")

def merge_results(json_strings):
    merged = {
        "critical_bugs": [],
        "dependency_issues": [],
        "security_risks": [],
        "performance_problems": [],
        "code_quality": [],
        "architecture_issues": [],
        "refactor_plan": [],
        "priority_fix_order": []
    }
    for js in json_strings:
        if not js.strip():
            continue
        try:
            data = json.loads(js)
        except json.JSONDecodeError:
            continue
        for key in merged:
            if key in data and isinstance(data[key], list):
                merged[key].extend(data[key])
    return merged

def build_token_aware_batches(structure, code_data, max_tokens_per_batch):
    base_prompt_no_files = build_prompt_strict(structure, {})
    base_tokens = estimate_tokens(base_prompt_no_files)
    available_tokens = max_tokens_per_batch - base_tokens
    if available_tokens <= 0:
        raise ValueError("max_tokens_per_batch too small for prompt overhead")

    batches = []
    current_batch = {}
    current_tokens = 0
    for path, content in code_data.items():
        file_block = f"\nFILE: {path}\n```\n{content}\n```"
        block_tokens = estimate_tokens(file_block)
        if current_tokens + block_tokens > available_tokens:
            if current_batch:
                batches.append(current_batch)
            current_batch = {}
            current_tokens = 0
        current_batch[path] = content
        current_tokens += block_tokens
    if current_batch:
        batches.append(current_batch)
    return batches

def process_batch(structure, batch_data):
    prompt = build_prompt_strict(structure, batch_data)
    for _ in range(2):
        try:
            return query_ollama(prompt)
        except Exception:
            continue
    return ""

def analyze_batches_parallel(structure, batches, num_parallel):
    results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        future_to_idx = {}
        for idx, batch_data in enumerate(batches):
            future = executor.submit(process_batch, structure, batch_data)
            future_to_idx[future] = idx
        with tqdm(total=len(batches), desc="Analyzing batches") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = json.dumps({"error": str(e)})
                results[idx] = result
                pbar.update(1)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--file", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS_PER_BATCH,
                        help="Max tokens per batch (including output reserve)")
    parser.add_argument("--parallel", type=int, default=OLLAMA_NUM_PARALLEL,
                        help="Number of parallel Ollama requests")
    parser.add_argument("--read-workers", type=int, default=8,
                        help="Number of threads for file reading")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)

    print("Scanning files...")
    file_paths = collect_files(root_dir, args.file)
    if not file_paths:
        print("No files found.")
        return

    print(f"Total files: {len(file_paths)}")
    code_data = read_files_parallel(file_paths, max_workers=args.read_workers)
    structure = get_structure(root_dir)

    max_input_tokens = args.max_tokens - OUTPUT_TOKEN_RESERVE
    if max_input_tokens <= 0:
        print("Error: max_tokens too low after output reserve.")
        return

    batches = build_token_aware_batches(structure, code_data, max_input_tokens)
    print(f"Created {len(batches)} batch(es) with token limit {max_input_tokens}")

    print("Running analysis...")
    batch_results = analyze_batches_parallel(structure, batches, args.parallel)

    final_result = merge_results(batch_results)

    print("\nFINAL OUTPUT\n")
    print(json.dumps(final_result, indent=2))

if __name__ == "__main__":
    main()