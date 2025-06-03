import os
import numpy as np
import pandas as pd
import yaml

import wandb


def flatten_yaml(yaml_object, parent_key="", sep="."):
    """
    Flatten a nested YAML file with a recursive function.
    The keys are concatenated to represent the hierarchy.
    """
    items = []
    for k, v in yaml_object.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_yaml(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(
                        flatten_yaml(item, f"{new_key}{sep}{i}", sep=sep).items()
                    )
                else:
                    items.append((f"{new_key}{sep}{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


def columns_non_relevant(df):
    columns_with_variability = []
    for column in df.columns:
        if all(isinstance(x, list) for x in df[column]):
            # Convert each list to a tuple for comparison since tuples are hashable
            unique_tuples = set(tuple(x) for x in df[column])
            if len(unique_tuples) == 1:
                columns_with_variability.append(column)
        else:
            if df[column].nunique() == 1:
                columns_with_variability.append(column)
    return columns_with_variability


def get_summary_runs(subset_name, exact_match=False, result_dir: str = None):
    # Collect all runs data

    acc = []

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    base_path = os.path.join(base_path, "results") if result_dir is None else os.path.join(base_path, result_dir)
    for group in os.listdir(base_path):
        group_path = os.path.join(base_path, group)
        if not os.path.isdir(group_path) or subset_name not in group:
            continue
        if subset_name != group and exact_match:
            continue

        for run in os.listdir(group_path):

            run_path = os.path.join(group_path, run)
            if not os.path.isdir(run_path):
                continue

            # from here should be the same as the wandb part
            log_path = os.path.join(run_path, "log_env.json")
            if not os.path.exists(log_path):
                continue
            run_path = os.path.join(run_path, ".hydra/config.yaml")
            flat_data = {}
            if os.path.exists(run_path):
                # Read the YAML file
                with open(run_path, "r") as file:
                    yaml_data = yaml.safe_load(file)

                # Flatten the YAML data
                flat_data = flatten_yaml(yaml_data)
                flat_data = {k: [v] for k, v in flat_data.items()}
            else:
                continue  # no wandb access!
            if "llm.path" in flat_data:
                acc.append(
                    pd.DataFrame(
                        {
                            "name": [run],
                            "group": [flat_data["llm.path"][0]],
                            "run_group": [group],
                            **flat_data,
                        },
                        index=[len(acc)],
                    )
                )
            else:
                acc.append(
                    pd.DataFrame(
                        {
                            "name": [run],
                            "group": [flat_data["llm1.path"][0]],
                            "run_group": [group],
                            **flat_data,
                        },
                        index=[len(acc)],
                    )
                )

    if len(acc) == 0:
        return pd.DataFrame(), pd.DataFrame()
    summary_df = pd.concat(acc)
    non_relevant = columns_non_relevant(summary_df)
    if len(summary_df) <= 1:
        non_relevant = []
    # remove "group" from non_relevant
    non_relevant = [
        c
        for c in non_relevant
        if c != "group"
        and c != "run_group"
        and c != "group_name"
        and c != "llm.path"
        and c != "llm1.path"
        and c != "llm2.path"
        and c != "llm.is_api"
        and c != "llm1.is_api"
        and c != "llm2.is_api"
        and c != "name"
    ]
    cols = [c for c in summary_df.columns if c.startswith("experiment.personas.")]
    cols.append("experiment.agent.name")
    cols.append("llm.top_p")
    cols.append("llm1.top_p")
    cols.append("llm2.top_p")
    cols.append("seed")
    cols.append("experiment.env.num_agents")
    cols.extend(non_relevant)
    summary_df = summary_df.drop(columns=cols, errors="ignore")

    summary_group_df = (
        summary_df.drop_duplicates(subset=["group"])
        .set_index("group", drop=True)
    )
    summary_group_df["id"] = summary_group_df.index

    llm_col = "llm.path" if "llm.path" in summary_group_df.columns else "llm1.path"

    summary_df.to_csv("summary.csv")
    summary_group_df.to_csv("summary_group.csv")
    return summary_df, summary_group_df


def load_runs_data(summary_df, summary_group_df, result_dir):
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    result_dir = os.path.join(base_path, "results") if result_dir is None else os.path.join(base_path, result_dir)
    run_data = {group: {} for group in summary_df["group"].unique()}
    for id, row in summary_df.iterrows():
        run_path = os.path.join(
            result_dir, row["run_group"], row["name"], "log_env.json"
        )
        df = pd.read_json(run_path, orient="records")
        is_list = df['agent_id'].apply(lambda x: isinstance(x, list))

        # Set first element of list for list rows only
        df.loc[is_list, 'agent_id'] = df.loc[is_list, 'agent_id'].map(lambda x: x[0])
        run_data[row["group"]][row["name"]] = df

    # print(f"Run data: {run_data}")

    return {
        "summary_group_df": summary_group_df,  # .to_dict(orient="records"),
        "summary_df": summary_df,
        "run_data": run_data,
    }

def load_runs_data_twoplayer(summary_df, summary_group_df, group_name, scenario, result_dir):
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    result_dir = os.path.join(base_path, "results") if result_dir is None else os.path.join(base_path, result_dir)
    run_data = {group: {} for group in summary_df["group"].unique()}
    for id, row in summary_df.iterrows():
        run_path = os.path.join(
            result_dir, row["run_group"], row["name"], "log_env.json"
        )

        df = pd.read_json(run_path, orient="records")

        df["model_1"] = row["llm1.path"] if "llm1.path" in row else row["llm.path"]
        df["model_2"] = row["llm2.path"] if "llm2.path" in row else scenario[2]
        is_survival = scenario[-1] == "survival" or scenario[-2] == "survival"
        df["group"] = group_name
        df["game"] = scenario[0]
        df["context"] = scenario[1]
        df["opponent"] = scenario[2]
        df["survival"] = True if is_survival else False
        df["prompting"] = "cot" if scenario[-1] == "cot" else None
        is_list = df['agent_id'].apply(lambda x: isinstance(x, list))

        # Set first element of list for list rows only
        df.loc[is_list, 'agent_id'] = df.loc[is_list, 'agent_id'].map(lambda x: x[0])
        run_data[row["group"]][row["name"]] = df

    return {
        "summary_group_df": summary_group_df,
        "summary_df": summary_df,
        "run_data": run_data,
    }