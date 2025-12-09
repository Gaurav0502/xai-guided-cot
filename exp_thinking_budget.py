"""
Experiment: Vary thinking budget
Tests: 0, 500, 1000, 2000, 4000
"""

from main import run_experiment

THINKING_BUDGETS = [500, 1000, 2000]

if __name__ == "__main__":
    for budget in THINKING_BUDGETS:
        print(f"\n{'='*50}")
        print(f"Running experiment: thinking_budget = {budget}")
        print("=" * 50)

        run_experiment(
            output_path=f"results/thinking_budget/budget_{budget}.json",
            thinking_budget=budget,
        )
