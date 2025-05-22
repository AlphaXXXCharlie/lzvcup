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
import json 

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ASP_MODEL_FILE = SCRIPT_DIR / "schedule_di.lp"
MATCH_RE = re.compile(r"match\((\d+),(\d+),(\d+)\)")

# --- Core Functions ---
def solve_with_api(asp_model_path, instance_file_path, timeout_seconds, clingo_options_str=None):
    """
    Solves the ASP problem using the Clingo API.
    Returns a dictionary with 'atoms_str', 'cost', 'optimality_proven', 'status'.
    """
    clingo_default_opts = [
        f"--opt-mode=optN",
        f"--solve-limit={timeout_seconds * 1000}"
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
        "status": "NO_SOLUTION_FOUND", # Default status
        "models_found": 0
    }
    
    print(f"[INFO] Starting Clingo solve process with options: {final_clingo_opts}...")
    current_model_count = 0
    try:
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                current_model_count += 1
                result["status"] = "MODEL_FOUND" # At least one model found
                result["optimality_proven"] = model.optimality_proven
                result["cost"] = list(model.cost) # Ensure it's a list
                
                current_model_atoms = list(model.symbols(shown=True))
                result["atoms_str"] = "\n".join(str(s) for s in current_model_atoms) # Keep the latest

                print(f"[DEBUG] Model {current_model_count}: Cost={model.cost}, Optimal={model.optimality_proven}, Atoms(sample): {current_model_atoms[:5]}...")
                if model.optimality_proven:
                    print(f"[INFO] Optimal model found by Clingo at model count {current_model_count}.")
                    

        result["models_found"] = current_model_count
        if current_model_count == 0: # No models yielded at all
             print("[WARNING] Clingo solve handle yielded no models.")
             result["status"] = "NO_SOLUTION_FOUND_HANDLE_EMPTY"


    except RuntimeError as e: # Catch Clingo runtime errors during solve
        print(f"[ERROR] Clingo solve process runtime error: {e}")
        result["status"] = "ERROR_SOLVE_RUNTIME"
        result["message"] = str(e)
        return result # Early exit on critical solve error

    if result["status"] == "MODEL_FOUND":
        print(f"[INFO] Clingo finished. {result['models_found']} model(s) processed.")
        print(f"[INFO] Final selected model - Cost: {result['cost']}, Optimal Proven: {result['optimality_proven']}")
    elif result["status"].startswith("NO_SOLUTION_FOUND"):
        timeout_occurred = timeout_seconds > 0 and not result["optimality_proven"] # Heuristic
        if timeout_occurred and ctl.statistics.get('summary', {}).get('exhausted', 0) == 0:
             result["status"] = "TIMEOUT_LIKELY"
             print(f"[WARNING] Clingo process finished. No optimal solution proven. Timeout ({timeout_seconds}s) likely occurred or problem is very hard.")
        else:
             print(f"[WARNING] Clingo process finished. Status: {result['status']}. Problem might be unsatisfiable or too complex.")
    
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

def compute_and_print_detailed_metrics(rounds_dict, num_teams, clingo_solve_result):
    print("\n--- Solution Quality Metrics ---")
    if clingo_solve_result["status"] not in ["MODEL_FOUND", "TIMEOUT_LIKELY"] or not rounds_dict:
        print("Metrics: N/A (No valid schedule data due to solver status or empty rounds).")
        print(f"Solver Status: {clingo_solve_result.get('status', 'Unknown')}")
        if "message" in clingo_solve_result: print(f"Message: {clingo_solve_result['message']}")
        return

    print(f"Clingo Optimality Proven: {clingo_solve_result['optimality_proven']}")
    print(f"Clingo Solution Cost (by priority levels): {clingo_solve_result['cost']}")
   

    violations_play_once = 0
    for r_num, matches_in_round in rounds_dict.items():
        team_play_counts = {t: 0 for t in range(1, num_teams + 1)}
        for h_team, a_team in matches_in_round:
            if 1 <= h_team <= num_teams: team_play_counts[h_team] += 1
            if 1 <= a_team <= num_teams: team_play_counts[a_team] += 1
        for team_id in range(1, num_teams + 1):
            if team_play_counts.get(team_id, 0) != 1: violations_play_once += 1
    print(f"Hard Constraint - Team plays once per round violations: {violations_play_once}")

    team_schedules_ha = {t: [] for t in range(1, num_teams + 1)} 
    for r_num in sorted(rounds_dict.keys()):
        for h_team, a_team in rounds_dict[r_num]:
            if 1 <= h_team <= num_teams: team_schedules_ha[h_team].append("H")
            if 1 <= a_team <= num_teams: team_schedules_ha[a_team].append("A")

    consecutive_home_2_count = 0
    consecutive_away_2_count = 0
    consecutive_home_3_count = 0
    consecutive_away_3_count = 0

    for team_id in range(1, num_teams + 1):
        seq = team_schedules_ha[team_id]
        for i in range(len(seq)):
            if i + 1 < len(seq) and seq[i] == seq[i+1]: # Found a pair
                if seq[i] == "H": consecutive_home_2_count += 1
                else: consecutive_away_2_count += 1
            if i + 2 < len(seq) and seq[i] == seq[i+1] == seq[i+2]: # Found a triplet
                if seq[i] == "H": consecutive_home_3_count += 1
                else: consecutive_away_3_count += 1
    
    print(f"Soft: Occurrences of exactly 2 consecutive home games: {consecutive_home_2_count - consecutive_home_3_count*2}") # Adjust because 3-in-a-row also counts as two 2-in-a-rows
    print(f"Soft: Occurrences of exactly 2 consecutive away games: {consecutive_away_2_count - consecutive_away_3_count*2}")
    print(f"Soft: Occurrences of 3+ consecutive home games: {consecutive_home_3_count}")
    print(f"Soft: Occurrences of 3+ consecutive away games: {consecutive_away_3_count}")
    
    total_imbalance = 0
    for team_id in range(1, num_teams + 1):
        h_total = team_schedules_ha[team_id].count("H")
        a_total = team_schedules_ha[team_id].count("A")
        total_imbalance += abs(h_total - a_total)
    print(f"Soft: Sum of absolute H/A imbalance over all teams: {total_imbalance}")
    print("Note: Metrics for 2-consecutive are approximate if 3+ exist; Clingo costs are the ground truth from the ASP.")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generates LZV Cup schedules using Clingo (DI Version).")
    parser.add_argument("-i", "--instances", nargs="+", required=True, help="Path to instance .lp file(s).")
    parser.add_argument("-o", "--output", required=True, help="Output directory for calendars.")
    parser.add_argument("-t", "--timeout", type=int, default=300, help="Clingo solver timeout (seconds).")
    parser.add_argument("--model", default=str(DEFAULT_ASP_MODEL_FILE), help=f"ASP model file (default: {DEFAULT_ASP_MODEL_FILE}).")
    parser.add_argument("--clingo-options", type=str, default="", help="Custom options to pass to Clingo (e.g., '--models=0 --opt-strategy=usc,pmres'). Quote if it contains spaces.")
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
            if clingo_solve_result.get("status") in ["MODEL_FOUND", "TIMEOUT_LIKELY"]: # TIMEOUT_LIKELY might have a suboptimal model
                rounds_data = parse_schedule_from_clingo_output(clingo_solve_result["atoms_str"])
            
            write_calendar_to_file(
                rounds_dict=rounds_data,
                output_dir_path=Path(args.output),
                instance_base_name=instance_base_name
            )
            
            compute_and_print_detailed_metrics(rounds_data, num_teams_for_metrics, clingo_solve_result)

        except Exception as e:
            print(f"[FATAL ERROR] Unhandled exception processing {instance_file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            
            if not clingo_solve_result: clingo_solve_result = {"status": "PYTHON_ERROR_PRE_SOLVE"}
            compute_and_print_detailed_metrics({}, num_teams_for_metrics, clingo_solve_result)


if __name__ == "__main__":
    main()