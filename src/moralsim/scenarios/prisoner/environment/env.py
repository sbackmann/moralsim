from omegaconf import DictConfig
from ..persona.cognition import utils as cognition_utils
from .utils import (
    prompt_opponent_action_report_production,
    prompt_opponent_action_report_base,
    prompt_opponent_action_report_privacy,
    prompt_opponent_action_report_venture,
    prompt_round_payoffs_production,
    prompt_round_payoffs_base,
    prompt_round_payoffs_privacy,
    prompt_round_payoffs_venture,
)
from moralsim.scenarios.common.environment import MoralityPerturbationEnv
import numpy as np

import logging

logger = logging.getLogger(__name__)


class PrisonerPerturbationEnv(MoralityPerturbationEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str], seed: int,
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name, seed)
        self.suckers_payoff_factor = 0.25

    def _prompt_round_payoffs(self, agent: str) -> str:
        chosen_action = self.internal_global_state["chosen_action"][agent]
        round_payoff = self.internal_global_state["round_payoff"][agent]
        both_defect = self.internal_global_state["both_defect"]
        input_total = self.internal_global_state["last_collected_resource"][agent]
        agent_name = self.agent_id_to_name[agent]
        if "pd_base" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_base
        elif "pd_privacy" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_privacy
        elif "pd_production" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_production
        elif "pd_venture" in self.cfg.name.lower():
            prompt_round_payoffs_fn = prompt_round_payoffs_venture
        else:
            raise ValueError(f"Unexpected environment name: {self.cfg.name}.")
        return prompt_round_payoffs_fn(chosen_action, round_payoff, both_defect, input_total, agent_name, self.min_payoff_to_survive, self.cfg.defect_base)
        
    
    def _prompt_opponent_action_report(self) -> str:
        if "pd_base" in self.cfg.name.lower():
            prompt_opponent_action_report_fn = prompt_opponent_action_report_base
        elif "pd_privacy" in self.cfg.name.lower():
            prompt_opponent_action_report_fn = prompt_opponent_action_report_privacy
        elif "pd_production" in self.cfg.name.lower():
            prompt_opponent_action_report_fn = prompt_opponent_action_report_production
        elif "pd_venture" in self.cfg.name.lower():
            prompt_opponent_action_report_fn = prompt_opponent_action_report_venture
        else:
            raise ValueError(f"Unexpected environment name: {self.cfg.name}.")
        return prompt_opponent_action_report_fn(self.agents, self.agent_id_to_name, self.internal_global_state["chosen_action"])

    def _agent_no_survival(self, agent: str) -> bool:
        return self.min_payoff_to_survive is not None and self.internal_global_state["round_payoff"][agent] < self.min_payoff_to_survive

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
        resource_per_agent = rs.randint(min_input, max_input)

        for agent in self.agents:
            res = resource_per_agent
            self.internal_global_state["collected_resource"][agent] += res
            self.internal_global_state["last_collected_resource"][agent] = res
    
    def _generate_inputs_low(self) -> None:
        """Low earnings: If other agent defects and you cooperate, you don't survive."""
        self._generate_inputs_stochastic(min_input=self.cfg.defect_base + 1, max_input=int(1 / self.suckers_payoff_factor) * self.min_payoff_to_survive - 1, seed_dec=self.perturbation_round)

    def _assign_resource(self) -> None:
        chosen_actions = self.internal_global_state["chosen_action"]
        # 1: Cooperate, 2: Defect
        input = self.internal_global_state["last_collected_resource"][self.agents[0]]
        equal_dist = False
        sum_actions = sum(list(chosen_actions.values()))
        if (both_defect := sum_actions == 2 * len(self.agents)):
            equal_dist = True
            input = self.cfg.defect_base
        elif sum_actions == len(self.agents):
            equal_dist = True
        self.internal_global_state["both_defect"] = both_defect
        for agent in self.agents:
            res = 0
            if equal_dist:
                res = input / len(self.agents)
            elif chosen_actions[agent] == 1:
                res = self.suckers_payoff_factor * input
            else:
                res = (1 - self.suckers_payoff_factor) * input
            
            self.log_round(self.internal_global_state["action"][agent], self.internal_global_state["last_collected_resource"][agent], res)

            self.internal_global_state["acc_payoff"][agent] += res
            self.internal_global_state["round_payoff"][agent] = res

            self.rewards[agent] += res
