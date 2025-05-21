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
    """
    Parses the raw Clingo output string to extract the schedule.
    Returns a dictionary of rounds, where each round is a list of (home, away) tuples.
    """
    rounds = {}
    for match_obj in MATCH_RE.finditer(clingo_output_str):
        r, h, a = map(int, match_obj.groups())
        rounds.setdefault(r, []).append((h, a))

    if not rounds and clingo_output_str.strip():
        print("[WARNING] parse_schedule_from_clingo_output found no 'match/3' atoms in non-empty Clingo output.")
    elif not rounds:
        print("[DEBUG] parse_schedule_from_clingo_output is returning an empty rounds dictionary (Clingo output might have been effectively empty of matches).")
    return rounds

def write_calendar_to_file(rounds_dict, output_dir_path, instance_base_name):
    """
    Writes the parsed schedule to a calendar file.
    """
    calendar_file = Path(output_dir_path) / f"{instance_base_name}_calendar.txt"

    print(f"[INFO] Writing calendar to: {calendar_file}")
    with open(calendar_file, "w") as f:
        if not rounds_dict:
            f.write("No schedule generated or no matches found.\n")
            print("[WARNING] Calendar file written with no schedule data because no rounds were parsed.")
            return

        for r_num in sorted(rounds_dict.keys()):
            # Sort matches within a round for consistent output, e.g., by home team
            sorted_matches = sorted(rounds_dict[r_num], key=lambda x: x[0])
            matches_str = ", ".join(f"{h}@{a}" for h, a in sorted_matches)
            f.write(f"Round {r_num}: {matches_str}\n")

def compute_and_print_metrics(rounds_dict, num_teams):
    """
    Computes and prints basic metrics for the generated schedule.
    """
    if not rounds_dict:
        print("[INFO] No schedule data to compute metrics for.")
        print("Violations (team plays/round): N/A (no schedule)")
        print("Consecutive home series: N/A, away series: N/A (no schedule)")
        return

    violations = 0
    for r_num, matches_in_round in rounds_dict.items():
        team_play_counts = {t: 0 for t in range(1, num_teams + 1)}
        for h_team, a_team in matches_in_round:
            # Added safety checks for team IDs being within the expected range
            if 1 <= h_team <= num_teams: team_play_counts[h_team] += 1
            if 1 <= a_team <= num_teams: team_play_counts[a_team] += 1

        for team_id in range(1, num_teams + 1):
            # Used .get for robustness if a team somehow isn't in counts (shouldn't happen with valid data, but good practice)
            if team_play_counts.get(team_id, 0) != 1:
                violations += 1
                print(f"[DEBUG Metric Violation] Team {team_id} plays {team_play_counts.get(team_id, 0)} times in round {r_num}.")
    print(f"Violations (each team should play once per round): {violations} instances")

    consecutive_home = 0
    consecutive_away = 0
    team_schedules_ha = {t: [] for t in range(1, num_teams + 1)}

    for r_num in sorted(rounds_dict.keys()):
        for h_team, a_team in rounds_dict[r_num]:
             # Added safety checks for team IDs being within the expected range
            if 1 <= h_team <= num_teams: team_schedules_ha[h_team].append("H")
            if 1 <= a_team <= num_teams: team_schedules_ha[a_team].append("A")

    for team_id in range(1, num_teams + 1):
        schedule_sequence = team_schedules_ha.get(team_id, []) # Used .get for robustness
        for i in range(len(schedule_sequence) - 1):
            if schedule_sequence[i] == "H" and schedule_sequence[i+1] == "H":
                consecutive_home += 1
            elif schedule_sequence[i] == "A" and schedule_sequence[i+1] == "A":
                consecutive_away += 1

    print(f"Consecutive home series: {consecutive_home}")
    print(f"Consecutive away series: {consecutive_away}")


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
        instance_base_name = instance_file_path.stem # Gets filename without extension
        print(f"\n--- Processing Instance: {instance_file_path.name} ---")

        # --- Start: Logic to parse 'n' from instance file ---
        num_teams_for_metrics = 16 # Default if n cannot be parsed
        try:
            with open(instance_file_path, 'r') as f_inst:
                content = f_inst.read()
                n_match = re.search(r"#const\s+n\s*=\s*(\d+)\.", content)
                if n_match:
                    num_teams_for_metrics = int(n_match.group(1))
                    print(f"[INFO] Parsed n={num_teams_for_metrics} from {instance_file_path.name} for Python-side metrics.")
                else:
                    print(f"[WARNING] Could not find '#const n = ...' in {instance_file_path.name}. Using default n={num_teams_for_metrics} for Python-side metrics.")
        except FileNotFoundError:
             # This case is already caught below, but good to handle parsing error specifically
             print(f"[ERROR] Instance file not found during n parsing: {instance_file_path.name}")
             # Keep default num_teams_for_metrics
        except Exception as e:
            print(f"[WARNING] Error reading or parsing n from {instance_file_path.name}: {e}. Using default n={num_teams_for_metrics} for Python-side metrics.")
        # --- End: Logic to parse 'n' ---


        try:
            clingo_raw_output = solve_with_api(
                asp_model_path=Path(args.model),
                instance_file_path=instance_file_path,
                timeout_seconds=args.timeout
            )

            # Check for effectively empty output, not just string == ""
            if not clingo_raw_output.strip() and not "No solution found" in clingo_raw_output : # Check if effectively empty
                 print("[WARNING] Clingo solve_with_api returned empty or no effective output. Calendar will likely be empty.")

            rounds_data = parse_schedule_from_clingo_output(clingo_raw_output)

            write_calendar_to_file(
                rounds_dict=rounds_data,
                output_dir_path=Path(args.output),
                instance_base_name=instance_base_name
            )

            # Pass the parsed num_teams to the metrics function
            compute_and_print_metrics(rounds_data, num_teams_for_metrics)

        except FileNotFoundError as e:
             print(f"[ERROR] A required file was not found: {e}")
        except RuntimeError as e: # Catch Clingo-specific runtime errors
            print(f"[ERROR] Clingo processing failed for {instance_file_path.name}: {e}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while processing {instance_file_path.name}: {e}")
            # Removed traceback for conciseness in this output, will add back in a polishing step
            # import traceback
            # traceback.print_exc()


if __name__ == "__main__":
    main()