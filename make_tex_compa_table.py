from pathlib import Path
from statistics import mean, pstdev


# Directory containing:
# playroom_case1_cloud/
# playroom_case1_mesh/
# playroom_case1_gs/
# ...
BASE_PATH = Path(
    "/home/user/Bureau/visual_navigation/IBVS_CODE/"
    "my_results/online_ibvs_test/playroom"
)

OUTPUT_PATH = BASE_PATH / "comparisons.tex"

CASE_NUMBERS = [1, 2, 3]

REPRESENTATIONS = {
    "Cloud": "cloud",
    "Mesh": "mesh",
    "3DGS": "gs"
}


# (results.txt key, LaTeX display name)
METRICS = [
    (
        "total_time",
        "Total time (min)"
    ),
    (
        "total_nbr_itrs",
        "IBVS iterations"
    ),
    (
        "avrg_condit_nbr",
        "Average condition number"
    ),
    (
        "R_diff_last_itr_deg",
        r"Final rotation error ($^\circ$)"
    ),
    (
        "t_diff_last_itr_norm",
        "Final normalized translation error"
    )
]


def load_results(results_path):
    """Load key-value results from one results.txt file."""

    results = {}

    with open(results_path, "r") as file:
        for line in file:
            line = line.strip()

            if "=" not in line:
                continue

            key, value = line.split("=", maxsplit=1)

            key = key.strip()
            value = value.strip()

            try:
                results[key] = float(value)
            except ValueError:
                results[key] = value

    return results


def validate_results(
    results,
    representation,
    case_number
):
    """Ensure that all required metrics exist."""

    for metric_key, _ in METRICS:
        if metric_key not in results:
            raise KeyError(
                f"Missing '{metric_key}' in "
                f"Playroom Scenario {case_number} "
                f"({representation})"
            )


def compute_results_statistics(all_results):
    """
    Calculate the mean and population standard deviation
    over the three scenarios.
    """

    statistics = {}

    for metric_key, _ in METRICS:
        values = [
            float(case_results[metric_key])
            for case_results in all_results
        ]

        statistics[metric_key] = {
            "mean": mean(values),
            "std": pstdev(values)
        }

    return statistics


def format_statistics(metric, values):
    """
    Format one mean ± standard-deviation result for LaTeX.
    """

    mean_value = values["mean"]
    std_value = values["std"]

    if metric == "total_time":
        # Original values are seconds; display minutes.
        mean_value /= 60.0
        std_value /= 60.0

        precision = 2

    elif metric == "total_nbr_itrs":
        precision = 1

    elif metric == "avrg_condit_nbr":
        precision = 2

    elif metric == "R_diff_last_itr_deg":
        precision = 3

    elif metric == "t_diff_last_itr_norm":
        precision = 3

    else:
        precision = 3

    return (
        f"${mean_value:.{precision}f} "
        rf"\pm {std_value:.{precision}f}$"
    )




def generate_comparison_table(statistics):
    """
    Generate a one-column-width LaTeX table comparing
    Cloud, Mesh and 3DGS.
    """

    rows = []

    for metric_key, metric_label in METRICS:
        cloud_value = format_statistics(
            metric_key,
            statistics["Cloud"][metric_key]
        )

        mesh_value = format_statistics(
            metric_key,
            statistics["Mesh"][metric_key]
        )

        gs_value = format_statistics(
            metric_key,
            statistics["3DGS"][metric_key]
        )

        rows.append(
            f"    {metric_label} & "
            f"{cloud_value} & "
            f"{mesh_value} & "
            f"{gs_value} \\\\"
        )

    rows_text = "\n".join(rows)

    return rf"""
\subsection{{Average Quantitative Results}}
\label{{app:playroom_average_results}}

\begingroup
\centering
\captionof{{table}}{{Average comparison over the three Playroom
scenarios (mean $\pm$ standard deviation).}}
\label{{tab:playroom_average_comparison}}

\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lccc@{{}}}}
    \toprule
    Metric & Cloud & Mesh & 3DGS \\
    \midrule
{rows_text}
    \bottomrule
\end{{tabular}}%
}}

\par
\endgroup
"""



def main():
    all_results = {
        representation: []
        for representation in REPRESENTATIONS
    }

    # Load Cloud, Mesh and 3DGS results for all three cases.
    for case_number in CASE_NUMBERS:
        for representation, suffix in REPRESENTATIONS.items():
            results_path = (
                BASE_PATH
                / f"playroom_case{case_number}_{suffix}"
                / "results"
                / "results.txt"
            )

            if not results_path.exists():
                raise FileNotFoundError(
                    f"Could not find: {results_path}"
                )

            results = load_results(
                results_path
            )

            validate_results(
                results,
                representation=representation,
                case_number=case_number
            )

            all_results[representation].append(
                results
            )

    # Calculate mean and standard deviation for each representation.
    statistics = {
        representation: compute_results_statistics(
            representation_results
        )
        for representation, representation_results
        in all_results.items()
    }

    # Generate only the LaTeX comparison table.
    latex_content = generate_comparison_table(
        statistics
    )

    with open(OUTPUT_PATH, "w") as file:
        file.write(latex_content)

    print(f"LaTeX table saved to: {OUTPUT_PATH}")

    # Print numerical statistics.
    for representation, representation_statistics in statistics.items():
        print(f"\n{representation} statistics:")

        for metric_key, values in representation_statistics.items():
            print(
                f"  {metric_key} = "
                f"{values['mean']:.6f} ± "
                f"{values['std']:.6f}"
            )

    print("\nGenerated LaTeX:")
    print(latex_content)


if __name__ == "__main__":
    main()