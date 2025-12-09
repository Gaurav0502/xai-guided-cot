"""
Experiment: Random selection vs Diverse selection (K-Means)
Should run Diverse selection first and then use the same number of samples.
"""

from main import run_experiment

RANDOM_SAMPLE_COUNTS = [20]

if __name__ == "__main__":
    # Random selection with different counts
    for n in RANDOM_SAMPLE_COUNTS:
        print(f"\n{'='*50}")
        print(f"Running experiment: Random selection (n={n})")
        print("=" * 50)

        run_experiment(
            output_path=f"results/selection/random_{n}.json",
            random_sample_count=n,
        )
