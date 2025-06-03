import pandas as pd
import os
from typing import Callable
from copy import deepcopy
from .preprocessing import get_summary_runs, load_runs_data, load_runs_data_twoplayer

def get_groups() -> list[str]:
    from itertools import product

    # Example inputs
    games = ["pg", "pd"]
    context = ["base", "privacy", "venture", "production"]
    opponents = ["dummy_cooperate", "dummy_defect"]
    survival = ["survival"]
    prompting = ["cot"]

    # Generate base combinations (without survival_risk)
    base_combos = list(product(games, context, opponents))
    survival_combos = [list(c) + [s] for c, s in product(base_combos, survival)]

    # Combine and convert base_combos to lists as well
    all_combos = [list(c) for c in base_combos] + survival_combos
    return [combo + prompting for combo in all_combos]


def load_all_scenario_results(with_binary: bool = False, transpose=False, scenario_dir=None):
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    if scenario_dir is None:
        scenario_dir = os.path.join(base_path, "notebooks/summary/scenarios")
    else:
        scenario_dir = os.path.join(base_path, scenario_dir)
    all_metrics = None
    scenario_names = []
    with os.scandir(scenario_dir) as it:
        for entry in it:
            if entry.is_file():
                continue
            scenario_name = entry.name
            scenario_names.append(scenario_name)
            tmp_df = pd.read_csv(os.path.join(entry.path, "metrics_summary.csv"))
            metric_columns = [col for col in tmp_df.columns[1:] if with_binary or "binary" not in col]
            if all_metrics is None:
                all_metrics = {column: {model: [] for model in tmp_df["model"]} for column in metric_columns}
            for column in metric_columns:
                for model in tmp_df["model"]:
                    all_metrics[column][model].append(tmp_df.loc[tmp_df["model"] == model, column].item())
    dfs = {metric: pd.DataFrame(metric_dict, index=scenario_names) for metric, metric_dict in all_metrics.items()}
    if transpose:
        dfs = {metric: df.T for metric, df in dfs.items()}
    combined_df = pd.DataFrame([])
    combined_df_std = pd.DataFrame([])
    for metric, df in dfs.items():
        if metric == "model":
            continue
        df_renamed = df.add_suffix(f"-{metric}")
        if "_std" in metric:
            combined_df_std = combined_df_std.join(df_renamed, how="outer")
        else:
            combined_df = combined_df.join(df_renamed, how="outer")
    dfs["combined"] = combined_df
    dfs["combined_std"] = combined_df_std
    return dfs

def apply_filter(dfs: dict[str, pd.DataFrame], filter_func: Callable, direction: str) -> dict[str, pd.DataFrame]:
    dfs_copy = deepcopy(dfs)
    for metric, df in dfs_copy.items():
        if direction == "row":
            dfs_copy[metric] = df.loc[filter_func(df.index), :].copy()
        else:
            dfs_copy[metric] = df.loc[:, filter_func(df.columns)].copy()
    return dfs_copy


def get_all_runs_model(model_long: str | list[str], scenarios: list[list[str]], result_dir: str = None) -> dict[str, pd.DataFrame]:
    run_data = {}
    if isinstance(model_long, str):
        model_long = [model_long]
    for scenario in scenarios:
        group_name = "_".join(scenario)
        summary_df, summary_group_df = get_summary_runs(group_name, exact_match=True)
        if summary_df.empty:
            continue
        data = load_runs_data(summary_df, summary_group_df, result_dir)["run_data"]
        for model in model_long:
            model_data = data.get(model, {})
            for run, run_details in model_data.items():
                is_survival = scenario[-1] == "survival" or scenario[-2] == "survival"
                run_details["group"] = group_name
                run_details["game"] = scenario[0]
                run_details["context"] = scenario[1]
                run_details["opponent"] = scenario[2]
                run_details["survival"] = True if is_survival else False
                run_details["prompting"] = scenario[-1] if scenario[-1].startswith("cot") else None
            run_data.update(**model_data)
    return run_data


def get_all_twoplayer_runs(result_dir: str = None) -> dict[str, pd.DataFrame]:
    scenarios = [
        ["pg", "production", "twoplayer", "cot"], ["pd", "production", "twoplayer", "cot"],
        ["pg", "production", "dummy_cooperate", "cot"], ["pd", "production", "dummy_cooperate", "cot"],
        ["pg", "production", "dummy_defect", "cot"], ["pd", "production", "dummy_defect", "cot"],
    ]
    run_data = {}
    for scenario in scenarios:
        group_name = "_".join(scenario)
        summary_df, summary_group_df = get_summary_runs(group_name, exact_match=True, result_dir=result_dir)
        if summary_df.empty:
            continue
        data = load_runs_data_twoplayer(summary_df, summary_group_df, group_name, scenario, result_dir)["run_data"]
        run_data.update({k: v for d in data.values() for k, v in d.items()})
    return run_data


def _get_groups_or_paraphrases(paraphrase: bool) -> list[list[str]]:
    if paraphrase:
        paraphrases = ["cot", "cot_p0", "cot_p1", "cot_p2"]
        configs = [["pd", "privacy", "dummy_defect", paraphrase] for paraphrase in paraphrases] + [["pg", "venture", "dummy_cooperate", "survival", paraphrase] for paraphrase in paraphrases]
    else:
        configs = get_groups()
    return configs

def convert_single_metric_df_to_latex(metrics_dict: dict[str, pd.DataFrame], std: bool, rename_models: bool = True, perc: bool = False, with_index: bool = True) -> dict[str, str]:

    res = {}
    for metric, df in metrics_dict.items():
        if "_std" in metric or "model" in metric:
            continue
        cols = df.columns
        if std:
            joint_df = df.join(metrics_dict[f"{metric}_std"], how="left", rsuffix="_std")
            if perc:
                joint_df = joint_df * 100
                suff = "\\%"
            else:
                suff = ""
            for col in cols:
                if col in ["game", "context", "opponent_type", "survival_type"]:
                    continue
                df[col] = joint_df.apply(
                    lambda row: f"{row[col]:.1f}"#{suff}" 
                                r"{\scriptsize ±"
                                f"{row[f'{col}_std']:.1f}"
                                "}",
                    axis=1
                )
        elif perc:
            df = df * 100
            df = df.applymap(lambda x: f"{x:.1f}")
        if rename_models:
            df = df.rename(columns=_replace_model_name)
            df.index = [_replace_model_name(index) for index in df.index]
        df.index = df.index.str.replace("_", "-")
        df.columns = df.columns.str.replace("_", "-")
        col_format = "l" + "c" * (len(df.columns) - 1)
        res.update({
            metric: df.to_latex(
                column_format=col_format,
                index=with_index,
                float_format="%.2f",
                label=f"metrics:{metric}",
                header=cols,
                escape=False,
            )
        })
    return res


def _replace_model_name(name:str) -> str:
    models_replace = {
        "gemini": "\gemini",
        "deepseek-v3": "\dsv",
        "deepseek-r1": "\dsr",
        "gpt-4o-mini": "\gptfouromini",
        "gpt-4o": "\gptfouro",
        "o3-mini": "\othreemini",
        "llama": "\llama",
        "qwen": "\qwen",
        "claude": "\claude",
    }
    sorted_items = sorted(models_replace.items(), key=lambda kv: -len(kv[0]))
    for key, macro in sorted_items:
        if key in name.lower():
            return macro
    return name