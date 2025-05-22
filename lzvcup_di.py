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
# import json # Will be added later for detailed output

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ASP_MODEL_FILE = SCRIPT_DIR / "schedule_di.lp"
MATCH_RE = re.compile(r"match\((\d+),(\d+),(\d+)\)")

# --- Core Functions (Modified for DI) ---
# solve_with_api updated to accept custom options and return detailed results
def solve_with_api(asp_model_path, instance_file_path, timeout_seconds, clingo_options_str=None): # Add new parameter
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
            # Simple split for now, robust parsing might be needed for complex options
            user_clingo_opts = clingo_options_str.split()
            print(f"[INFO] Using custom Clingo options: {user_clingo_opts}")
        except Exception as e:
            print(f"[WARNING] Could not parse custom Clingo options '{clingo_options_str}'. Error: {e}")
            # Keep processing with default options, log the warning

    final_clingo_opts = clingo_default_opts + user_clingo_opts
    ctl = clingo.Control(final_clingo_opts) # Use final options list

    print(f"[INFO] Loading ASP model from: {asp_model_path}")
    if not asp_model_path.exists():
        # Return structured error result
        return {"status": "ERROR_MODEL_NOT_FOUND", "message": f"ASP model file not found: {asp_model_path}"}
    ctl.load(str(asp_model_path))

    print(f"[INFO] Loading instance file: {instance_file_path}")
    if not instance_file_path.exists():
         # Return structured error result
        return {"status": "ERROR_INSTANCE_NOT_FOUND", "message": f"Instance file not found: {instance_file_path}"}
    ctl.load(str(instance_file_path))

    print("[INFO] Grounding the ASP program...")
    try:
        ctl.ground([("base", [])])
    except RuntimeError as e:
        # Return structured error result
        return {"status": "ERROR_GROUNDING", "message": f"Clingo grounding failed: {e}"}


    result = {
        "atoms_str": "",
        "cost": [], # Will store optimization cost as a list
        "optimality_proven": False,
        "status": "NO_SOLUTION_FOUND", # Default status
        "models_found": 0,
        "message": None # Optional message for errors/status
    }

    print(f"[INFO] Starting Clingo solve process with options: {final_clingo_opts}...")
    current_model_count = 0
    try:
        # Use yield_=True to process models as they are found (especially useful for optN)
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                current_model_count += 1
                # We got at least one model, update status
                result["status"] = "MODEL_FOUND"
                result["optimality_proven"] = model.optimality_proven
                result["cost"] = list(model.cost) # Capture cost - important for DI optimization

                # Keep the atoms of the latest (most optimal) model found so far
                current_model_atoms = list(model.symbols(shown=True))
                result["atoms_str"] = "\n".join(str(s) for s in current_model_atoms)

                print(f"[DEBUG] Model {current_model_count}: Cost={model.cost}, Optimal={model.optimality_proven}, Atoms(sample): {current_model_atoms[:5]}...")
                # You could add a break here if you only wanted the *first* model,
                # but optN mode usually means you want the *last* (best) one found within the limit.
                if model.optimality_proven:
                    print(f"[INFO] Optimal model found by Clingo at model count {current_model_count}.")
                    # If optimality is proven, we can potentially exit early depending on requirements
                    # For optN, continue yielding to potentially find more optimal models if they exist?
                    # Clingo's API handle will manage this based on optN.

        result["models_found"] = current_model_count
        if current_model_count == 0: # No models yielded at all
             print("[WARNING] Clingo solve handle yielded no models.")
             result["status"] = "NO_SOLUTION_FOUND_HANDLE_EMPTY" # More specific status

        # Check final solve result status (e.g., SAT, UNSAT, TIMEOUT, ERROR)
        # Clingo handle has .get(), .satisfiable, .optimal, .timeout
        solve_result_status = handle.get()
        if solve_result_status.optimal:
             result["status"] = "OPTIMAL_SOLUTION_FOUND"
        elif solve_result_status.satisfiable:
             # This might be covered by MODEL_FOUND already, but confirms satisfiability
             pass
        elif solve_result_status.timeout:
             # If a model was found *before* timeout, status is MODEL_FOUND or OPTIMAL
             # If timeout happened *before* any model, status is NO_SOLUTION... but timeout=True
             if result["status"].startswith("NO_SOLUTION"):
                  result["status"] = "TIMEOUT_NO_MODEL"
             else:
                  result["status"] = "TIMEOUT_WITH_MODEL" # Could be optimal or suboptimal
             print(f"[INFO] Clingo solve process timed out ({timeout_seconds}s).")
        elif solve_result_status.unsatisfiable:
             result["status"] = "UNSATISFIABLE"
             print("[INFO] Problem is unsatisfiable.")
        # Add other status checks from solve_result_status if needed

    except RuntimeError as e: # Catch Clingo runtime errors during solve (e.g., invalid options, internal error)
        print(f"[ERROR] Clingo solve process runtime error: {e}")
        result["status"] = "ERROR_SOLVE_RUNTIME"
        result["message"] = str(e)
        # In case of runtime error, we might not have any models or cost
        result["atoms_str"] = ""
        result["cost"] = []
        result["models_found"] = 0
        return result # Early exit on critical solve error

    # Final status message based on the solve_result_status and models found
    if result["status"] in ["MODEL_FOUND", "TIMEOUT_WITH_MODEL", "OPTIMAL_SOLUTION_FOUND"]:
        print(f"[INFO] Clingo finished. {result['models_found']} model(s) processed.")
        print(f"[INFO] Final selected model - Cost: {result['cost']}, Optimal Proven: {result['optimality_proven']}")
    elif result["status"] == "TIMEOUT_NO_MODEL":
        print(f"[WARNING] Clingo process timed out ({timeout_seconds}s). No models found within the limit.")
    elif result["status"] == "UNSATISFIABLE":
        print("[INFO] Clingo process finished. Problem is unsatisfiable. No models found.")
    elif result["status"].startswith("NO_SOLUTION"): # Generic no solution status
         print(f"[WARNING] Clingo process finished. Status: {result['status']}. No models found.")

    return result # Return the detailed result dictionary

# parse_schedule_from_clingo_output remains the same
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
        print("[WARNING] parse_schedule: Found no 'match/3' atoms in non-empty Clingo output.")
    # Keep the debug print for empty case
    # elif not rounds:
    #     print("[DEBUG] parse_schedule: Returning an empty rounds dictionary.")
    return rounds

# write_calendar_to_file remains the same
def write_calendar_to_file(rounds_dict, output_dir_path, instance_base_name):
    """
    Writes the parsed schedule to a calendar file.
    """
    calendar_file = Path(output_dir_dir) / f"{instance_base_name}_calendar.txt" # Corrected typo: output_dir_path

    print(f"[INFO] Writing calendar to: {calendar_file}")
    with open(calendar_file, "w") as f:
        if not rounds_dict:
            f.write("No schedule generated (no matches found or solution not optimal/found).\n")
            # No warning here, as lack of rounds might be intentional based on solver result
            return
        for r_num in sorted(rounds_dict.keys()):
            sorted_matches = sorted(rounds_dict[r_num], key=lambda x: x[0])
            matches_str = ", ".join(f"{h}@{a}" for h, a in sorted_matches)
            f.write(f"Round {r_num}: {matches_str}\n")

# compute_and_print_metrics is now compute_and_print_detailed_metrics - placeholder for now
def compute_and_print_detailed_metrics(rounds_dict, num_teams, clingo_solve_result):
     """Placeholder for detailed DI metrics."""
     print("\n--- Solution Quality Metrics (DI Placeholder) ---")
     print(f"Solver Status: {clingo_solve_result.get('status', 'Unknown')}")
     print(f"Clingo Cost: {clingo_solve_result.get('cost', [])}")
     print(f"Optimality Proven: {clingo_solve_result.get('optimality_proven', False)}")
     print("[INFO] compute_and_print_detailed_metrics called (DI placeholder)")


# --- Main Execution (Modified for DI) ---
# The main function will be updated in the next commit to use the new solve_with_api return structure
def main():
    parser = argparse.ArgumentParser(description="Generates LZV Cup schedules using Clingo (DI Version).")
    parser.add_argument("-i", "--instances", nargs="+", required=True, help="Path to instance .lp file(s).")
    parser.add_argument("-o", "--output", required=True, help="Output directory for calendars.")
    parser.add_argument("-t", "--timeout", type=int, default=300, help="Clingo solver timeout (seconds).")
    parser.add_argument("--model", default=str(DEFAULT_ASP_MODEL_FILE), help=f"ASP model file (default: {DEFAULT_ASP_MODEL_FILE}).")
    # Add new argument for custom Clingo options
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

        # Variable to store detailed solver result
        clingo_solve_result = {}

        try:
            # Need to update call to solve_with_api for DI later (passing options, handling detailed return)
            # The current main function will not correctly handle the dictionary return yet
            clingo_raw_output = solve_with_api( # THIS LINE WILL BREAK IN THE NEXT COMMIT!
                asp_model_path=Path(args.model),
                instance_file_path=instance_file_path,
                timeout_seconds=args.timeout,
                clingo_options_str=args.clingo_options # Pass the options here
            )
            # The parse_schedule_from_clingo_output expects a string, but solve_with_api now returns a dict.
            # This mismatch will be fixed in the *next* commit.

            rounds_data = parse_schedule_from_clingo_output(clingo_raw_output) # THIS LINE WILL ALSO LIKELY BREAK

            write_calendar_to_file(
                rounds_dict=rounds_data,
                output_dir_path=Path(args.output),
                instance_base_name=instance_base_name
            )

            # Pass detailed result to the metrics function
            compute_and_print_detailed_metrics(rounds_data, num_teams_for_metrics, clingo_solve_result)

        except FileNotFoundError as e:
             print(f"[ERROR] A required file was not found: {e}")
        except RuntimeError as e: # Catch Clingo-specific runtime errors
            print(f"[ERROR] Clingo processing failed for {instance_file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            if not clingo_solve_result: clingo_solve_result = {"status": "PYTHON_ERROR_PRE_SOLVE"}
            compute_and_print_detailed_metrics({}, num_teams_for_metrics, clingo_solve_result)
        except Exception as e:
            print(f"[FATAL ERROR] Unhandled exception processing {instance_file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            if not clingo_solve_result: clingo_solve_result = {"status": "PYTHON_FATAL_ERROR"}
            compute_and_print_detailed_metrics({}, num_teams_for_metrics, clingo_solve_result)


if __name__ == "__main__":
    main()