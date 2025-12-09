"""
Experiment: Vary num_examples_per_agent
Tests: 3, 5, 10, 15, 20, -1 (special case, single agent with all examples)
"""

from main import run_experiment

NUM_EXAMPLES_VALUES = [5, 10, 20, -1]

if __name__ == "__main__":
    for n in NUM_EXAMPLES_VALUES:
        label = "all" if n == -1 else n
        print(f"\n{'='*50}")
        print(f"Running experiment: num_examples_per_agent = {label}")
        print("=" * 50)

        run_experiment(
            output_path=f"results/num_examples/n_{label}.json",
            num_examples_per_agent=n,
        )
