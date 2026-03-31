import os
import requests
import argparse
from pathlib import Path
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
BATCH_SIZE = 12


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


def read_files(file_paths):
    data = {}
    for path in tqdm(file_paths, desc="Reading files"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if len(content) > MAX_FILE_SIZE:
                    content = content[:MAX_FILE_SIZE] + "\nTRUNCATED"
                data[path] = content
        except Exception as e:
            data[path] = f"ERROR: {str(e)}"
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
  "refactor_plan": [
    "step 1",
    "step 2"
  ],
  "priority_fix_order": [
    "1",
    "2"
  ]
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
            "stream": False
        }
    )
    if response.status_code != 200:
        raise Exception(response.text)
    return response.json()["response"]


def batch_analyze(structure, code_data):
    paths = list(code_data.keys())
    results = []
    batches = [paths[i:i + BATCH_SIZE] for i in range(0, len(paths), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Analyzing batches"):
        batch_data = {p: code_data[p] for p in batch}
        prompt = build_prompt(structure, batch_data)
        result = query_ollama(prompt)
        results.append(result)

    return "\n\n".join(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--file", default=None)
    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)

    print("Scanning files")
    file_paths = collect_files(root_dir, args.file)

    if not file_paths:
        print("No files found")
        return

    print(f"Total files: {len(file_paths)}")

    code_data = read_files(file_paths)
    structure = get_structure(root_dir)

    print("Running analysis")
    result = batch_analyze(structure, code_data)

    print("\nFINAL OUTPUT\n")
    print(result)


if __name__ == "__main__":
    main()