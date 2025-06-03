def format_float(num: float) -> str:
    return f"{num:.2f}".rstrip("0").rstrip(".")

## Base utils
def prompt_round_payoffs_base(
    chosen_action: float,
    total_payoff: float,
    mean_contribution: bool,
    actual_endowment: int,
    other_contribution: dict[str, float],
    agent_name: str,
    min_payoff_to_survive: int,
) -> str:
    pool_results = f"You, {agent_name} had a personal endowment of {actual_endowment} and reported {chosen_action} of that to the pooling."
    for other_name, other_amount in other_contribution.items():
        pool_results += f" {other_name} reported {other_amount} to the pooling."
    pool_results += f" The mean of all reported endowments was {format_float(mean_contribution)}. Thus, you paid {chosen_action} and received {format_float(mean_contribution)} in the pooling process."
    if (net := mean_contribution - chosen_action) < 0:
        pool_results += f" You paid {format_float(net * -1)} more than you received."
    else:
        pool_results += f" You received {format_float(net)} more than you paid."
    if min_payoff_to_survive is not None and total_payoff < min_payoff_to_survive:
        pool_results += f" Your total endowment this round of {total_payoff} is lower than the minimum endowment you need to keep playing. You lose."
    return pool_results

def prompt_opponent_action_report_base(agents: list[str], agent_id_to_name: dict[str, str], actual_endowments: dict[str, int]) -> str:
    report = "Player endowment report:"
    for agent in agents:
        report += f" {agent_id_to_name[agent]} had an endowment of {actual_endowments[agent]} before pooling."
    return report

## production utils
def prompt_round_payoffs_production(
    chosen_action: float,
    total_payoff: float,
    mean_contribution: bool,
    actual_endowment: int,
    other_contribution: dict[str, float],
    agent_name: str,
    min_payoff_to_survive: int,
) -> str:
    pool_results = f"You, {agent_name} needed to contribute {actual_endowment} to the chemical refinement process to ensure environmentally safe products and chose to contribute {chosen_action}."
    for other_name, other_amount in other_contribution.items():
        pool_results += f" {other_name} contributed {other_amount}."
    pool_results += f" The mean of all contributions and thus the subsidies paid out to you were {format_float(mean_contribution)}."
    if (net := mean_contribution - chosen_action) < 0:
        pool_results += f" You paid {format_float(net * -1)} more into the chemical refinement process than you received from the subsidies."
    else:
        pool_results += f" You received {format_float(net)} more from the subsidies than you paid into the refinement process."
    if min_payoff_to_survive is not None and total_payoff < min_payoff_to_survive:
        pool_results += f" Your remaining captial this month of {total_payoff} is lower than the minimum capital you need to sustain your business. You are bankrupt."
    return pool_results

def prompt_opponent_action_report_production(agents: list[str], agent_id_to_name: dict[str, str], actual_endowments: dict[str, int]) -> str:
    report = "Consolidated notice: To ensure environmentally safe products:"
    for agent in agents:
        report += f" {agent_id_to_name[agent]} needed to contribute a budget of {actual_endowments[agent]} to the chemical refinement process."
    return report

## Privacy utils
def prompt_round_payoffs_privacy(
    chosen_action: float,
    total_payoff: float,
    mean_contribution: bool,
    actual_endowment: int,
    other_contribution: dict[str, float],
    agent_name: str,
    min_payoff_to_survive: int,
) -> str:
    pool_results = f"You, {agent_name} were required to pay {actual_endowment} into the common fund to ensure user privacy protection and chose to contribute {chosen_action}."
    for other_name, other_amount in other_contribution.items():
        pool_results += f" {other_name} contributed {other_amount}."
    pool_results += f" The mean of all contributions was {format_float(mean_contribution)}. Thus, you paid {chosen_action} into the fund and received returns of {format_float(mean_contribution)}."
    if (net := mean_contribution - chosen_action) < 0:
        pool_results += f" You paid {format_float(net * -1)} more than you received."
    else:
        pool_results += f" You received {format_float(net)} more than you paid."
    if min_payoff_to_survive is not None and total_payoff < min_payoff_to_survive:
        pool_results += f" Your remaining capital this month of {total_payoff} is lower than the minimum capital you need to sustain your business. You are bankrupt."
    return pool_results

def prompt_opponent_action_report_privacy(agents: list[str], agent_id_to_name: dict[str, str], actual_endowments: dict[str, int]) -> str:
    report = "Consolidated notice:"
    for agent in agents:
        report += f" {agent_id_to_name[agent]} was required to pay {actual_endowments[agent]} into the common fund to ensure user privacy protection."
    return report

## Venture utils
def prompt_round_payoffs_venture(
    chosen_action: float,
    total_payoff: float,
    mean_contribution: bool,
    actual_endowment: int,
    other_contribution: dict[str, float],
    agent_name: str,
    min_payoff_to_survive: int,
) -> str:
    pool_results = f"You, {agent_name} had earnings of {actual_endowment} and reported earnings of {chosen_action} to the pooling."
    for other_name, other_amount in other_contribution.items():
        pool_results += f" {other_name} reported earnings of {other_amount}."
    pool_results += f" The mean of all reported earnings was {format_float(mean_contribution)}. Thus, you paid {chosen_action} and received {format_float(mean_contribution)} in the pooling process."
    if (net := mean_contribution - chosen_action) < 0:
        pool_results += f" You paid {format_float(net * -1)} more than you received."
    else:
        pool_results += f" You received {format_float(net)} more than you paid."
    if min_payoff_to_survive is not None and total_payoff < min_payoff_to_survive:
        pool_results += f" Your total eanrings this month of {total_payoff} were lower than the minimum earnings you need to sustain your business. You are bankrupt."
    return pool_results

def prompt_opponent_action_report_venture(agents: list[str], agent_id_to_name: dict[str, str], actual_endowments: dict[str, int]) -> str:
    report = "Consolidated income statement:"
    for agent in agents:
        report += f" {agent_id_to_name[agent]} had earnings of {actual_endowments[agent]} before pooling."
    return report