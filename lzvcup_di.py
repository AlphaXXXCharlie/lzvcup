#!/usr/bin/env python3
"""
lzvcup_di.py — Enhanced In‑process Clingo scheduling for LZV Cup (DI Version).
Supports custom Clingo options and detailed cost/metric reporting.
"""

import argparse
import re
import os
import sys
import clingo
from pathlib import Path
import traceback
import json

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ASP_MODEL_FILE = SCRIPT_DIR / "schedule_di.lp"
MATCH_RE = re.compile(r"match\((\d+),(\d+),(\d+)\)")

# --- Core Functions ---
def solve_with_api(asp_model_path, instance_file_path, timeout_seconds, clingo_options_str=None):
    """
    Solves the ASP problem using the Clingo API.
    Returns a dictionary with 'atoms_str', 'cost', 'optimality_proven', 'status', and 'message' if applicable.
    """
    clingo_default_opts = [
        f"--opt-mode=optN", # Find a sequence of optimal models
        f"--solve-limit={timeout_seconds * 1000}" # Clingo timeout is in milliseconds
    ]

    user_clingo_opts = []
    if clingo_options_str:
        try:
            user_clingo_opts = clingo_options_str.split()
            print(f"[INFO] Using custom Clingo options: {user_clingo_opts}")
        except Exception as e:
            print(f"[WARNING] Could not parse custom Clingo options '{clingo_options_str}'. Error: {e}")

    final_clingo_opts = clingo_default_opts + user_clingo_opts
    ctl = clingo.Control(final_clingo_opts)

    print(f"[INFO] Loading ASP model from: {asp_model_path}")
    if not asp_model_path.exists():
        return {"status": "ERROR_MODEL_NOT_FOUND", "message": f"ASP model file not found: {asp_model_path}"}
    ctl.load(str(asp_model_path))

    print(f"[INFO] Loading instance file: {instance_file_path}")
    if not instance_file_path.exists():
        return {"status": "ERROR_INSTANCE_NOT_FOUND", "message": f"Instance file not found: {instance_file_path}"}
    ctl.load(str(instance_file_path))

    print("[INFO] Grounding the ASP program...")
    try:
        ctl.ground([("base", [])])
    except RuntimeError as e:
        return {"status": "ERROR_GROUNDING", "message": f"Clingo grounding failed: {e}"}

    result = {
        "atoms_str": "",
        "cost": [],
        "optimality_proven": False,
        "status": "NO_SOLUTION_FOUND",
        "models_found": 0,
        "message": None
    }

    print(f"[INFO] Starting Clingo solve process with options: {final_clingo_opts}...")
    current_model_count = 0
    try:
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                current_model_count += 1
                result["status"] = "MODEL_FOUND"
                result["optimality_proven"] = model.optimality_proven
                result["cost"] = list(model.cost)

                current_model_atoms = list(model.symbols(shown=True))
                result["atoms_str"] = "\n".join(str(s) for s in current_model_atoms)

                print(f"[DEBUG] Model {current_model_count}: Cost={model.cost}, Optimal={model.optimality_proven}, Atoms(sample): {current_model_atoms[:5]}...")
                if model.optimality_proven:
                    print(f"[INFO] Optimal model found by Clingo at model count {current_model_count}.")


        result["models_found"] = current_model_count
        if current_model_count == 0:
             print("[WARNING] Clingo solve handle yielded no models.")
             result["status"] = "NO_SOLUTION_FOUND_HANDLE_EMPTY"

        solve_result_status = handle.get()
        if solve_result_status.optimal:
             result["status"] = "OPTIMAL_SOLUTION_FOUND"
        elif solve_result_status.timeout:
             if result["status"].startswith("NO_SOLUTION"):
                  result["status"] = "TIMEOUT_NO_MODEL"
             else:
                  result["status"] = "TIMEOUT_WITH_MODEL"
             print(f"[INFO] Clingo solve process timed out ({timeout_seconds}s).")
        elif solve_result_status.unsatisfiable:
             result["status"] = "UNSATISFIABLE"
             print("[INFO] Problem is unsatisfiable.")


    except RuntimeError as e:
        print(f"[ERROR] Clingo solve process runtime error: {e}")
        result["status"] = "ERROR_SOLVE_RUNTIME"
        result["message"] = str(e)
        result["atoms_str"] = ""
        result["cost"] = []
        result["models_found"] = 0
        return result

    if result["status"] in ["MODEL_FOUND", "TIMEOUT_WITH_MODEL", "OPTIMAL_SOLUTION_FOUND"]:
        print(f"[INFO] Clingo finished. {result['models_found']} model(s) processed.")
        print(f"[INFO] Final selected model - Cost: {result['cost']}, Optimal Proven: {result['optimality_proven']}")
    elif result["status"] == "TIMEOUT_NO_MODEL":
        print(f"[WARNING] Clingo process timed out ({timeout_seconds}s). No models found within the limit.")
    elif result["status"] == "UNSATISFIABLE":
        print("[INFO] Clingo process finished. Problem is unsatisfiable. No models found.")
    elif result["status"].startswith("NO_SOLUTION"):
         print(f"[WARNING] Clingo process finished. Status: {result['status']}. No models found.")

    return result

def parse_schedule_from_clingo_output(clingo_output_str):
    rounds = {}
    for match_obj in MATCH_RE.finditer(clingo_output_str):
        r, h, a = map(int, match_obj.groups())
        rounds.setdefault(r, []).append((h, a))

    if not rounds and clingo_output_str.strip():
        print("[WARNING] parse_schedule: Found no 'match/3' atoms in non-empty Clingo output.")
    return rounds

def write_calendar_to_file(rounds_dict, output_dir_path, instance_base_name):
    calendar_file = Path(output_dir_path) / f"{instance_base_name}_calendar.txt"

    print(f"[INFO] Writing calendar to: {calendar_file}")
    with open(calendar_file, "w") as f:
        if not rounds_dict:
            f.write("No schedule generated (no matches found or solution not optimal/found).\n")
            return
        for r_num in sorted(rounds_dict.keys()):
            sorted_matches = sorted(rounds_dict[r_num], key=lambda x: x[0])
            matches_str = ", ".join(f"{h}@{a}" for h, a in sorted_matches)
            f.write(f"Round {r_num}: {matches_str}\n")

# compute_and_print_metrics is now compute_and_print_detailed_metrics - MODIFIED TO RETURN METRICS
def compute_and_print_detailed_metrics(rounds_dict, num_teams, clingo_solve_result):
    """
    Computes and prints detailed metrics for the generated schedule.
    Also reports Clingo solver outcome information.
    Returns a dictionary of computed metrics.
    """
    # --- Start: Prepare dictionary to return ---
    metrics = {
        "violations_play_once": 0,
        "consecutive_home_2": 0,
        "consecutive_away_2": 0,
        "consecutive_home_3": 0,
        "consecutive_away_3": 0,
        "total_imbalance": 0
    }
    # --- End: Prepare dictionary to return ---


    print("\n--- Solution Quality Metrics ---")
    if clingo_solve_result.get("status") not in ["MODEL_FOUND", "TIMEOUT_WITH_MODEL", "OPTIMAL_SOLUTION_FOUND"]:
        print("Metrics: N/A (No valid schedule data due to solver status).")
        print(f"Solver Status: {clingo_solve_result.get('status', 'Unknown')}")
        if "message" in clingo_solve_result and clingo_solve_result["message"]: print(f"Message: {clingo_solve_result['message']}")
        if clingo_solve_result.get("status") in ["MODEL_FOUND", "TIMEOUT_WITH_MODEL", "OPTIMAL_SOLUTION_FOUND"] and not rounds_dict:
             print("[WARNING] Solver reported a model status but no schedule data was parsed.")
        return metrics # Return the initialized metrics dictionary on failure


    print(f"Clingo Optimality Proven: {clingo_solve_result.get('optimality_proven', False)}")
    print(f"Clingo Solution Cost (by priority levels): {clingo_solve_result.get('cost', [])}")

    # Hard Constraint - Team plays once per round violations
    violations_play_once = 0 # Calculate for printing
    for r_num, matches_in_round in rounds_dict.items():
        team_play_counts = {t: 0 for t in range(1, num_teams + 1)}
        for h_team, a_team in matches_in_round:
            if 1 <= h_team <= num_teams: team_play_counts[h_team] += 1
            if 1 <= a_team <= num_teams: team_play_counts[a_team] += 1
        for team_id in range(1, num_teams + 1):
            if team_play_counts.get(team_id, 0) != 1:
                violations_play_once += 1
    metrics["violations_play_once"] = violations_play_once # Store in metrics dict
    print(f"Hard Constraint - Team plays once per round violations: {metrics['violations_play_once']} instances")


    # Refined Consecutive Game Counting (Exact 2 and 3+)
    consecutive_home_2_count_temp = 0
    consecutive_away_2_count_temp = 0
    consecutive_home_3_count_triplets = 0
    consecutive_away_3_count_triplets = 0

    team_schedules_ha = {t: [] for t in range(1, num_teams + 1)}

    for r_num in sorted(rounds_dict.keys()):
        for h_team, a_team in rounds_dict[r_num]:
            if 1 <= h_team <= num_teams: team_schedules_ha[h_team].append("H")
            if 1 <= a_team <= num_teams: team_schedules_ha[a_team].append("A")

    for team_id in range(1, num_teams + 1):
        seq = team_schedules_ha.get(team_id, [])
        for i in range(len(seq)):
            if i + 1 < len(seq) and seq[i] == seq[i+1]:
                if seq[i] == "H": consecutive_home_2_count_temp += 1
                else: consecutive_away_2_count_temp += 1
            if i + 2 < len(seq) and seq[i] == seq[i+1] == seq[i+2]:
                if seq[i] == "H": consecutive_home_3_count_triplets += 1
                else: consecutive_away_3_count_triplets += 1

    # Store refined counts in metrics dict
    metrics["consecutive_home_3"] = consecutive_home_3_count_triplets
    metrics["consecutive_away_3"] = consecutive_away_3_count_triplets
    # Calculate and store exact 2 counts
    metrics["consecutive_home_2"] = consecutive_home_2_count_temp - metrics["consecutive_home_3"] * 2
    metrics["consecutive_away_2"] = consecutive_away_2_count_temp - metrics["consecutive_away_3"] * 2


    print(f"Soft: Occurrences of exactly 2 consecutive home games: {metrics['consecutive_home_2']}")
    print(f"Soft: Occurrences of exactly 2 consecutive away games: {metrics['consecutive_away_2']}")
    print(f"Soft: Occurrences of 3+ consecutive home games: {metrics['consecutive_home_3']}")
    print(f"Soft: Occurrences of 3+ consecutive away games: {metrics['consecutive_away_3']}")

    # H/A Balance Metric
    total_imbalance = 0
    for team_id in range(1, num_teams + 1):
        seq = team_schedules_ha.get(team_id, [])
        h_total = seq.count("H")
        a_total = seq.count("A")
        total_imbalance += abs(h_total - a_total)
    metrics["total_imbalance"] = total_imbalance # Store in metrics dict
    print(f"Soft: Sum of absolute H/A imbalance over all teams: {metrics['total_imbalance']}")

    print("Note: Metrics for 2-consecutive are approximate if 3+ exist; Clingo costs are the ground truth from the ASP.")

    return metrics # --- Return the computed metrics dictionary ---


# --- Main Execution ---
# The main function will be updated in the next commit to capture and use the returned metrics
def main():
    parser = argparse.ArgumentParser(description="Generates LZV Cup schedules using Clingo (DI Version).")
    parser.add_argument("-i", "--instances", nargs="+", required=True, help="Path to instance .lp file(s).")
    parser.add_argument("-o", "--output", required=True, help="Output directory for calendars.")
    parser.add_argument("-t", "--timeout", type=int, default=300, help="Clingo solver timeout (seconds).")
    parser.add_argument("--model", default=str(DEFAULT_ASP_MODEL_FILE), help=f"ASP model file (default: {DEFAULT_ASP_MODEL_FILE}).")
    parser.add_argument("--clingo-options", type=str, default="", help="Custom options to pass to Clingo (e.g., '--models=0 --opt-strategy=usc,pmres'). Quote if it contains spaces.")
    parser.add_argument("--json-output", action="store_true", help="Output detailed results in JSON format.")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    for instance_file_str in args.instances:
        instance_file_path = Path(instance_file_str)
        instance_base_name = instance_file_path.stem
        print(f"\n--- Processing Instance: {instance_file_path.name} ---")

        num_teams_for_metrics = 16
        try:
            with open(instance_file_path, 'r') as f_inst:
                content = f_inst.read()
                n_match = re.search(r"#const\s+n\s*=\s*(\d+)\.", content)
                if n_match: num_teams_for_metrics = int(n_match.group(1))
            print(f"[INFO] Parsed n={num_teams_for_metrics} from {instance_file_path.name} for Python metrics.")
        except Exception as e:
            print(f"[WARNING] Error reading n: {e}. Using default n={num_teams_for_metrics} for metrics.")

        clingo_solve_result = {}

        try:
            clingo_solve_result = solve_with_api(
                asp_model_path=Path(args.model),
                instance_file_path=instance_file_path,
                timeout_seconds=args.timeout,
                clingo_options_str=args.clingo_options
            )

            rounds_data = {}
            if clingo_solve_result.get("atoms_str"):
                 rounds_data = parse_schedule_from_clingo_output(clingo_solve_result["atoms_str"])

            # Call metrics function to print to console (it now also returns values)
            # *** This line will be updated in the next commit to capture the return value ***
            compute_and_print_detailed_metrics(rounds_data, num_teams_for_metrics, clingo_solve_result)


            # Handle JSON output flag (still using placeholders for metrics)
            if args.json_output:
                 json_output_file = Path(args.output) / f"{instance_base_name}_results.json"
                 print(f"[INFO] Writing detailed results to JSON file: {json_output_file}")
                 detailed_results_dict = {
                     "instance": instance_file_path.name,
                     "num_teams": num_teams_for_metrics,
                     "solver_status": clingo_solve_result.get("status", "Unknown"),
                     "optimality_proven": clingo_solve_result.get("optimality_proven", False),
                     "clingo_cost": clingo_solve_result.get("cost", []),
                     "python_metrics": { # Still placeholder values
                         "violations_play_once": "TODO: Get value from metrics func",
                         "consecutive_home_2": "TODO: Get value from metrics func",
                         "consecutive_away_2": "TODO: Get value from metrics func",
                         "consecutive_home_3": "TODO: Get value from metrics func",
                         "consecutive_away_3": "TODO: Get value from metrics func",
                         "total_imbalance": "TODO: Get value from metrics func"
                     },
                     "calendar_file": f"{instance_base_name}_calendar.txt" if rounds_data and clingo_solve_result.get("status") in ["MODEL_FOUND", "TIMEOUT_WITH_MODEL", "OPTIMAL_SOLUTION_FOUND"] else None,
                     "error_message": clingo_solve_result.get("message")
                 }
                 with open(json_output_file, "w") as f:
                     json.dump(detailed_results_dict, f, indent=4)


            if rounds_data and clingo_solve_result.get("status") in ["MODEL_FOUND", "TIMEOUT_WITH_MODEL", "OPTIMAL_SOLUTION_FOUND"]:
                 write_calendar_to_file(
                     rounds_dict=rounds_data,
                     output_dir_path=Path(args.output),
                     instance_base_name=instance_base_name
                 )
            elif not rounds_data and clingo_solve_result.get("status") not in ["ERROR_GROUNDING", "ERROR_SOLVE_RUNTIME"]:
                 print("[INFO] No schedule data to write to calendar file.")


        except FileNotFoundError as e:
             print(f"[ERROR] A required file was not found: {e}")
             err_result = {"status": "PYTHON_ERROR_FILE_NOT_FOUND", "message": str(e)}
             compute_and_print_detailed_metrics({}, num_teams_for_metrics, err_result) # Call metrics (it returns now)
             if args.json_output:
                 json_output_file = Path(args.output) / f"{instance_base_name}_results.json"
                 # Need to call metrics function again to get the empty dict on error for JSON
                 error_metrics = compute_and_print_detailed_metrics({}, num_teams_for_metrics, err_result) # This will print N/A again, bit redundant
                 error_results_dict = {
                     "instance": instance_file_path.name,
                     "num_teams": num_teams_for_metrics,
                     "solver_status": err_result["status"],
                     "error_message": err_result["message"],
                     "optimality_proven": False,
                     "clingo_cost": [],
                     "python_metrics": error_metrics, # Use the returned empty metrics
                     "calendar_file": None
                 }
                 print(f"[INFO] Writing error details to JSON file: {json_output_file}")
                 with open(json_output_file, "w") as f:
                     json.dump(error_results_dict, f, indent=4)

        except RuntimeError as e:
            print(f"[ERROR] Clingo processing failed for {instance_file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            # clingo_solve_result should be populated by solve_with_api
            compute_and_print_detailed_metrics({}, num_teams_for_metrics, clingo_solve_result) # Call metrics (it returns now)
            if args.json_output:
                 json_output_file = Path(args.output) / f"{instance_base_name}_results.json"
                 # Need to call metrics again to get the empty dict on error for JSON
                 error_metrics = compute_and_print_detailed_metrics({}, num_teams_for_metrics, clingo_solve_result) # Redundant print
                 error_results_dict = {
                     "instance": instance_file_path.name,
                     "num_teams": num_teams_for_metrics,
                     "solver_status": clingo_solve_result.get("status", "CLINGO_RUNTIME_ERROR"),
                     "error_message": clingo_solve_result.get("message", str(e)),
                     "optimality_proven": clingo_solve_result.get("optimality_proven", False),
                     "clingo_cost": clingo_solve_result.get("cost", []),
                     "python_metrics": error_metrics, # Use the returned empty metrics
                     "calendar_file": None
                 }
                 print(f"[INFO] Writing error details to JSON file: {json_output_file}")
                 with open(json_output_file, "w") as f:
                     json.dump(error_results_dict, f, indent=4)

        except Exception as e:
            print(f"[FATAL ERROR] Unhandled exception while processing {instance_file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            if not clingo_solve_result: clingo_solve_result = {"status": "PYTHON_FATAL_ERROR"}
            compute_and_print_detailed_metrics({}, num_teams_for_metrics, clingo_solve_result) # Call metrics (it returns now)
            if args.json_output:
                 json_output_file = Path(args.output) / f"{instance_base_name}_results.json"
                 # Need to call metrics again to get the empty dict on error for JSON
                 error_metrics = compute_and_print_detailed_metrics({}, num_teams_for_metrics, clingo_solve_result) # Redundant print
                 error_results_dict = {
                     "instance": instance_file_path.name,
                     "num_teams": num_teams_for_metrics,
                     "solver_status": clingo_solve_result.get("status", "PYTHON_FATAL_ERROR"),
                     "error_message": str(e),
                     "optimality_proven": clingo_solve_result.get("optimality_proven", False),
                     "clingo_cost": clingo_solve_result.get("cost", []),
                     "python_metrics": error_metrics, # Use the returned empty metrics
                     "calendar_file": None
                 }
                 print(f"[INFO] Writing error details to JSON file: {json_output_file}")
                 with open(json_output_file, "w") as f:
                     json.dump(error_results_dict, f, indent=4)


if __name__ == "__main__":
    main()