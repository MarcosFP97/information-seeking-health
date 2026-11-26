"""Verify exact McNemar test results for the Llama3 reproduction."""

import csv
from math import comb, isclose
from pathlib import Path

SUMMARY_FILE = (
    Path(__file__).resolve().parent
    / "outputs"
    / "llama3"
    / "analysis"
    / "summary.csv"
)

ALPHA = 0.05


def exact_mcnemar(b, c):
    """Calculate the exact two-sided McNemar p-value."""
    n = b + c

    if n == 0:
        return 1.0

    return min(
        1.0,
        2 * sum(
            comb(n, k) * (0.5 ** n)
            for k in range(min(b, c) + 1)
        )
    )


def main():
    with SUMMARY_FILE.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            prompt = row["prompt"]

            b = int(row["original_only_correct"])
            c = int(row["reproduction_only_correct"])

            p_value = exact_mcnemar(b, c)
            significant = p_value < ALPHA

            stored_p = float(row["mcnemar_p_value"])

            verified = isclose(
                p_value, stored_p, abs_tol=1e-9
            )

            print(f"\nPrompt: {prompt}")
            print(f"Original only correct: {b}")
            print(f"Reproduction only correct: {c}")
            print(f"P-value: {p_value:.4f}")
            print(f"Statistically significant: {significant}")
            print(f"Stored p-value verified: {verified}")


if __name__ == "__main__":
    main()
