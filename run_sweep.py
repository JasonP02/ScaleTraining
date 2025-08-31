#!/usr/bin/env python3
"""
Helper script to initialize and run wandb sweeps.
"""
import subprocess
import sys

def init_sweep():
    """Initialize the wandb sweep."""
    print("Initializing wandb sweep...")
    result = subprocess.run(["wandb", "sweep", "sweep_config.yaml"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Sweep initialized successfully!")
        print(result.stdout)
        # Extract sweep ID from output - look for various patterns
        sweep_id = None
        for line in result.stdout.split('\n'):
            if 'wandb agent' in line:
                # Extract the ID which is usually the last part
                parts = line.split()
                for part in parts:
                    if '/' in part and not part.startswith('http'):
                        sweep_id = part
                        break
            elif 'sweep ID:' in line.lower():
                sweep_id = line.split()[-1]
        
        if sweep_id:
            print(f"\n" + "="*50)
            print(f"SWEEP ID: {sweep_id}")
            print(f"To run the sweep, use:")
            print(f"wandb agent {sweep_id}")
            print("="*50)
            return sweep_id
        else:
            print("Could not extract sweep ID from output.")
            return None
    else:
        print("Error initializing sweep:")
        print(result.stderr)
        return None

def run_agent(sweep_id):
    """Run the wandb agent."""
    print(f"Running wandb agent for sweep: {sweep_id}")
    subprocess.run(["wandb", "agent", sweep_id])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        if len(sys.argv) > 2:
            run_agent(sys.argv[2])
        else:
            print("Please provide sweep ID: python run_sweep.py run <sweep-id>")
    else:
        init_sweep()