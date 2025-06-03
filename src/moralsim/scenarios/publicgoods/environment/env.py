from ..persona.cognition import utils as cognition_utils
from .utils import (
    prompt_opponent_action_report_production,
    prompt_opponent_action_report_privacy,
    prompt_opponent_action_report_base,
    prompt_opponent_action_report_venture,
    prompt_round_payoffs_production,
    prompt_round_payoffs_privacy,
    prompt_round_payoffs_base,
    prompt_round_payoffs_venture
)
from moralsim.scenarios.common.environment import MoralityPerturbationEnv
import numpy as np

import logging

logger = logging.getLogger(__name__)


class PublicGoodsPerturbationEnv(MoralityPerturbationEnv):

    def _prompt_round_payoffs(self, agent: str) -> str:
        chosen_action = self.internal_global_state["chosen_action"][agent]
        pool_amount = self.internal_global_state["round_payoff"][agent]
        mean_contribution = self.internal_global_state["mean_reported_earnings"]
        actual_endowment = self.internal_global_state["last_collected_resource"][agent]
        other_contribution = {self.agent_id_to_name[other_agent]: self.internal_global_state["chosen_action"][other_agent] for other_agent in self.agents if other_agent != agent}
        agent_name = self.agent_id_to_name[agent]
        if "pg_base" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_base
        elif "pg_production" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_production
        elif "pg_privacy" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_privacy
        elif "pg_venture" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_venture
        else:
            raise ValueError(f"Unexpected environment name: {self.cfg.name}.")
        return prompt_round_payoffs_fn(chosen_action, pool_amount, mean_contribution, actual_endowment, other_contribution, agent_name, self.min_payoff_to_survive)
    
    def _prompt_opponent_action_report(self) -> str:
        if "pg_base" in self.cfg.name.lower():
            prompt_opponent_action_report_fn = prompt_opponent_action_report_base
        elif "pg_production" in self.cfg.name.lower():
            prompt_opponent_action_report_fn = prompt_opponent_action_report_production
        elif "pg_privacy" in self.cfg.name.lower():
            prompt_opponent_action_report_fn = prompt_opponent_action_report_privacy
        elif "pg_venture" in self.cfg.name.lower():
            prompt_opponent_action_report_fn = prompt_opponent_action_report_venture
        else:
            raise ValueError(f"Unexpected environment name: {self.cfg.name}.")
        return prompt_opponent_action_report_fn(self.agents, self.agent_id_to_name, self.internal_global_state["last_collected_resource"])

    def _agent_no_survival(self, agent: str) -> bool:
        total_earnings = self.internal_global_state["last_collected_resource"][agent] - self.internal_global_state["chosen_action"][agent] + self.internal_global_state["mean_reported_earnings"]
        return self.min_payoff_to_survive is not None and total_earnings < self.min_payoff_to_survive

    def _apply_perturbations(self) -> None:
        if self.perturbation is None:
            return
        if self.num_round == self.perturbation.round:
            if self.perturbation.type == "low_inputs_season":
                self._generate_inputs = self._generate_inputs_low
                self.perturbation_round = self.perturbation.round
                cognition_utils.MIN_PAYOFF_TO_SURVIVE = self.perturbation.min_payoff_to_survive
                self.min_payoff_to_survive = self.perturbation.min_payoff_to_survive
            logger.info(f"Apply perturbation: {self.perturbation.type}")
    
    def _generate_inputs_stochastic(self, *, min_input=None, max_input=None, seed_dec=0) -> None:
        if min_input is None:
            min_input = self.cfg.min_input
            max_input = self.cfg.max_input + 1 # inclusive
        rs = np.random.RandomState(self.seed * 1000 + self.num_round - seed_dec) # different earnings for each seed
        resource_per_agent = {agent: rs.randint(min_input, max_input) for agent in self.agents}

        for agent in self.agents:
            res = resource_per_agent[agent]
            self.internal_global_state["collected_resource"][agent] += res
            self.internal_global_state["last_collected_resource"][agent] = res
            self.rewards[agent] += res
    
    def _generate_inputs_low(self) -> None:
        """Low earnings: If other agent defects and you cooperate, you don't survive."""
        self._generate_inputs_stochastic(min_input=self.min_payoff_to_survive, max_input=self.cfg.min_input, seed_dec=self.perturbation_round)

    def _assign_resource(self) -> None:
        reported_amounts = self.internal_global_state["chosen_action"]
        
        mean = np.mean([reported_amount for agent, reported_amount in reported_amounts.items() if agent in self.agents])
        self.internal_global_state["mean_reported_earnings"] = mean

        for agent in self.agents:
            res = mean - reported_amounts[agent]
            action = self.internal_global_state["action"][agent]
            actual_earnings = self.internal_global_state["last_collected_resource"][agent]
            total_payoff = actual_earnings + res
            self.log_round(action, actual_earnings, total_payoff)

            self.internal_global_state["collected_resource"][agent] += res
            self.internal_global_state["acc_payoff"][agent] += total_payoff
            self.internal_global_state["round_payoff"][agent] = total_payoff

            self.rewards[agent] += res
