#!/usr/bin/env python3
"""
lzvcup.py — In‐process Clingo scheduling for LZV Cup.
Requires: pip install clingo
"""

import argparse
import re
import os
import sys
import clingo
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ASP_MODEL_FILE = SCRIPT_DIR / "schedule.lp"
MATCH_RE = re.compile(r"match\((\d+),(\d+),(\d+)\)")

# --- Core Functions (Placeholders) ---

# --- Core Functions ---
def solve_with_api(asp_model_path, instance_file_path, timeout_seconds):
    """
    Solves the ASP problem using the Clingo API.
    Returns the raw string output of the best model's shown atoms.
    """
    ctl = clingo.Control([
        f"--opt-mode=optN", # Find a sequence of optimal models
        f"--solve-limit={timeout_seconds * 1000}" # Clingo timeout
    ])

    print(f"[INFO] Loading ASP model from: {asp_model_path}")
    if not asp_model_path.exists():
        print(f"[ERROR] ASP model file not found at: {asp_model_path}")
        raise FileNotFoundError(f"ASP model file not found: {asp_model_path}")
    ctl.load(str(asp_model_path))

    print(f"[INFO] Loading instance file: {instance_file_path}")
    if not instance_file_path.exists():
        print(f"[ERROR] Instance file not found at: {instance_file_path}")
        raise FileNotFoundError(f"Instance file not found: {instance_file_path}")
    ctl.load(str(instance_file_path))

    print("[INFO] Grounding the ASP program...")
    ctl.ground([("base", [])])

    best_model_atoms_str = ""
    best_model_found = False

    print("[INFO] Starting Clingo solve process...")
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            best_model_found = True
            print(f"[DEBUG] Clingo found a model. Optimality: {model.optimality_proven}, Atoms: {model.symbols(shown=True)}")
            # Keep the atoms of the latest (most optimal) model
            current_model_atoms = list(model.symbols(shown=True))
            best_model_atoms_str = "\n".join(str(s) for s in current_model_atoms)

    if not best_model_found:
        print("[ERROR] Clingo reported no solution (no models yielded).")

        return ""

    print(f"[DEBUG] Raw output string from best model for parsing:\n---\n{best_model_atoms_str}\n---")
    return best_model_atoms_str

def parse_schedule_from_clingo_output(clingo_output_str):
    """Placeholder for parsing."""
    print("[INFO] parse_schedule_from_clingo_output called (placeholder)")
    return {}

def write_calendar_to_file(rounds_dict, output_dir_path, instance_base_name):
    """Placeholder for writing."""
    print("[INFO] write_calendar_to_file called (placeholder)")

def compute_and_print_metrics(rounds_dict, num_teams):
    """Placeholder for metrics."""
    print("[INFO] compute_and_print_metrics called (placeholder)")

# --- Main Execution (Minimal) ---
def main():
    parser = argparse.ArgumentParser(description="Generates sports schedules using Clingo.")
    parser.add_argument(
        "-i", "--instances", nargs="+", required=True,
        help="Path to one or more instance .lp files (e.g., instance_04.lp)."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the directory where output calendars will be saved."
    )
    parser.add_argument(
        "-t", "--timeout", type=int, default=300,
        help="Timeout in seconds for the Clingo solver (default: 300)."
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_ASP_MODEL_FILE),
        help=f"Path to the ASP model file (default: {DEFAULT_ASP_MODEL_FILE})."
    )
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    for instance_file_str in args.instances:
        instance_file_path = Path(instance_file_str)
        instance_base_name = instance_file_path.stem
        print(f"\n--- Processing Instance: {instance_file_path.name} ---")

        # Placeholder calls
        clingo_raw_output = solve_with_api(Path(args.model), instance_file_path, args.timeout)
        rounds_data = parse_schedule_from_clingo_output(clingo_raw_output)
        write_calendar_to_file(rounds_data, Path(args.output), instance_base_name)
        compute_and_print_metrics(rounds_data, 0) # num_teams is placeholder

if __name__ == "__main__":
    main()