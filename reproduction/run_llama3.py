import os
import sys
import time
import json
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from importlib.metadata import version

# Locate repository root
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

# Import the original research functions
from src.llm.llama3 import load_answers, load_context, ask

MODEL = "llama3.1:8b-instruct-q8_0"
YEAR = 2022
TEMPERATURE = 0
NUM_CTX = 8192

PROMPTS = {
    "no-context": "",
    "non-expert": "nonexpert",
    "expert": "expert"
}

PAPER_ACCURACY = {
    "no-context": 0.82,
    "non-expert": 0.80,
    "expert": 0.80
}


def run_command(command):
    return subprocess.check_output(
        command, text=True
    ).strip()


def run_experiment(prompt_type):

    context_name = PROMPTS[prompt_type]

    # Create a unique directory for each run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output_dir = (
        REPO_DIR / "reproduction" / "outputs" /
        "llama3" / prompt_type / run_id
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["OUTDIR"] = str(output_dir)

    # Load the original dataset and prompt
    answers = load_answers(YEAR)
    context = load_context(context_name)

    assert len(answers) == 50, "Expected 50 questions"

    log = {
        "date": datetime.now().astimezone().isoformat(),
        "mac_model": run_command(
            ["sysctl", "-n", "hw.model"]
        ),
        "chip": run_command(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ),
        "ram_gib": int(run_command(
            ["sysctl", "-n", "hw.memsize"]
        )) / (1024 ** 3),
        "macos_version": platform.mac_ver()[0],
        "python_version": platform.python_version(),
        "ollama_version": run_command(
            ["ollama", "--version"]
        ),
        "ollama_python_package": version("ollama"),
        "git_commit": run_command(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"]
        ),
        "git_status": run_command(
            ["git", "-C", str(REPO_DIR), "status", "--short"]
        ),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "num_ctx": NUM_CTX,
        "prompt_type": prompt_type,
        "context_text": context,
        "few_shot_examples": 0,
        "dataset": "TREC HM 2022",
        "number_questions": len(answers),
        "paper_accuracy": PAPER_ACCURACY[prompt_type],
        "changes": [
            "Corrected context and dataset paths",
            "Changed output directory",
            "Added overwrite protection",
            "Used repository prompt wording",
            "Added external runtime and metadata logging"
        ],
        "errors": None
    }

    print("\nStarting experiment:", prompt_type)
    print("Model:", MODEL)
    print("Questions:", len(answers))
    print("Output directory:", output_dir)

    start = time.perf_counter()

    try:
        accuracy = ask(
            MODEL,
            answers,
            context,
            context_name,
            YEAR
        )

        log["accuracy"] = accuracy
        log["correct_answers"] = round(
            accuracy * len(answers)
        )
        log["incorrect_answers"] = (
            len(answers) - log["correct_answers"]
        )
        log["difference_percentage_points"] = round(
            (accuracy - PAPER_ACCURACY[prompt_type]) * 100,
            2
        )

    except Exception as error:
        log["errors"] = str(error)
        raise

    finally:
        elapsed = time.perf_counter() - start

        log["total_runtime_seconds"] = round(elapsed, 4)

        if log["errors"] is None:
            log["time_per_question_seconds"] = round(
                elapsed / len(answers), 4
            )

        with open(output_dir / "run_log.json", "x") as f:
            json.dump(log, f, indent=4)

    print("\n--- Experiment Results ---")
    print(f"Prompt: {prompt_type}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Paper accuracy: {PAPER_ACCURACY[prompt_type]:.2%}")
    print(f"Difference: {log['difference_percentage_points']} pp")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print(f"Time per question: {elapsed / len(answers):.2f} seconds")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "prompt",
        choices=list(PROMPTS.keys())
    )

    args = parser.parse_args()
    run_experiment(args.prompt)