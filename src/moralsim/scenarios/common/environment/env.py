from omegaconf import DictConfig
from typing import Callable

from .obs import ActionObs
from moralsim.persona.common import PersonaEvent, PersonaAction, PersonaActionChoice
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pettingzoo.utils import AgentSelector
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
import logging
import wandb

logger = logging.getLogger(__name__)


class MoralityPerturbationEnv(ABC):

    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str], seed: int,
    ) -> None:
        self.cfg = cfg
        self.experiment_storage = experiment_storage

        self.possible_agents = [f"persona_{i}" for i in range(5)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_id_to_name = map_id_to_name

        self.LOCATION = "office"
        self.seed = seed
        self.perturbation = cfg.perturbations[0].perturbation if len(cfg.perturbations) > 0 else None
        self._generate_inputs: Callable = self._generate_inputs_stochastic
        self.min_payoff_to_survive = cfg.min_payoff_to_survive

    def save_log(self) -> None:
        log_file = f"{self.experiment_storage}/log_env.json"
        pd.concat(self.df_acc).to_json(
           log_file , orient="records"
        )
        wandb.save(log_file)

    def log_round(
        self,
        action: PersonaActionChoice,
        round_input_value: int,
        round_payoff: float,
    ) -> None:
        tmp = {
            "agent_id": [action.agent_id],
            "round": [self.num_round],
            "action": ["choose_action"],
            "round_input_value": [round_input_value],
            "chosen_action": [action.quantity],
            "round_payoff": [round_payoff],
            "html_interactions": [action.html_interactions],
        }
        df_log = pd.DataFrame(tmp, index=[len(self.df_acc)])
        self.df_acc.append(df_log)

    def _get_reflection_day(self, current_date: datetime) -> datetime:
        next_month = current_date.replace(day=28) + timedelta(days=4)
        last_day_of_current_month = next_month - timedelta(days=next_month.day)
        return last_day_of_current_month

    def _get_expiration_month(self, current_date: datetime) -> datetime:
        target_month = current_date + relativedelta(months=self.cfg.event_expiration_months + 1)
        last_day_of_target_month = target_month - relativedelta(days=target_month.day)
        return last_day_of_target_month

    def _observe_pre_round(self, agent: str) -> ActionObs:
        if self._agent_selector.is_first():
            self._generate_inputs()
        obs = ActionObs(
            phase=self.phase,
            current_location=self.internal_global_state["next_location"][agent],
            current_location_agents=self.internal_global_state["next_location"],
            current_time=self.internal_global_state["next_time"][agent],
            events=[],
            context="",
            agent_resource_num=self.internal_global_state["last_collected_resource"][agent],
        )
        return obs

    def _observe_office_after_round(self, agent: str) -> ActionObs:
        obs = ActionObs(
            phase=self.phase,
            current_location=self.internal_global_state["next_location"][agent],
            current_location_agents=self.internal_global_state["next_location"],
            current_time=self.internal_global_state["next_time"][agent],
            events=[
                PersonaEvent(
                    self._prompt_round_payoffs(agent),
                    created=self.internal_global_state["next_time"][agent] + relativedelta(days=1),
                    expiration=self._get_expiration_month(
                        self.internal_global_state["next_time"][agent]
                    ),
                    always_include=True,
                ),
                PersonaEvent(
                    self._prompt_opponent_action_report(),
                    created=self._get_reflection_day(
                        self.internal_global_state["next_time"][agent]
                    ) - timedelta(days=1)
                    - relativedelta(minutes=1),
                    expiration=self._get_expiration_month(
                        self.internal_global_state["next_time"][agent]
                    ),
                    always_include=True,
                ),
            ],
            context="",
            agent_resource_num={agent: 0 for agent in self.agents},
        )
        return obs

    def _observe_home(self, agent) -> ActionObs:
        events = []
        state = ActionObs(
            phase=self.phase,
            current_location=self.internal_global_state["next_location"][agent],
            current_location_agents=self.internal_global_state["next_location"],
            current_time=self.internal_global_state["next_time"][agent],
            events=events,
            context="",
            agent_resource_num={agent: 0 for agent in self.agents},
        )
        return state
    
    def _observe(self, agent) -> ActionObs:
        """
        Observe should return the observation of the specified agent.

        Depending on the current phase, the observation will be different
        """

        if self.phase == self.LOCATION:
            state = self._observe_pre_round(agent)
        elif self.phase == "office_after_round":
            state = self._observe_office_after_round(agent)
        elif self.phase == "home":
            state = self._observe_home(agent)
        return state

    def _init_agent(self, agent: str) -> None:
        self.internal_global_state["collected_resource"][agent] = 0
        self.internal_global_state["chosen_action"][agent] = 0
        self.internal_global_state["last_collected_resource"][agent] = 0
        self.internal_global_state["next_location"][agent] = self.LOCATION
        self.internal_global_state["next_time"][agent] = datetime(2024, 1, 1, 1, 0, 0)

        self.rewards[agent] = 0.0
        self.terminations[agent] = False
        if "acc_payoff" not in self.internal_global_state.keys():
            self.internal_global_state["acc_payoff"] = {}
            self.internal_global_state["round_payoff"] = {}
        self.internal_global_state["acc_payoff"][agent] = 0
        self.internal_global_state["round_payoff"][agent] = 0


    def _step_pre_round(self, action: PersonaActionChoice):
        res = action.quantity
        self.internal_global_state["chosen_action"][self.agent_selection] = res
        self.internal_global_state["action"][self.agent_selection] = action
        self.internal_global_state["next_location"][
            self.agent_selection
        ] = self.LOCATION
        if self._agent_selector.is_last():
            self._assign_resource()
            self.phase = self._phase_selector.next()
        self.agent_selection = self._agent_selector.next()

    def _step_office_after_round(self, action: PersonaActionChoice):

        self.internal_global_state["next_location"][self.agent_selection] = "home"
        self.internal_global_state["next_time"][self.agent_selection] = (
            self._get_reflection_day(
                self.internal_global_state["next_time"][self.agent_selection]
            )
        )
        if self._agent_selector.is_last():
            self.phase = self._phase_selector.next()
        self.agent_selection = self._agent_selector.next()
    
    def _step_home(self, action: PersonaAction) -> None:

        self.internal_global_state["next_location"][
            self.agent_selection
        ] = self.LOCATION
        self.internal_global_state["next_time"][self.agent_selection] += timedelta(
            days=1
        )
        agent, agent_type = action.agent_id
        if agent_type != "dummy" and self._agent_no_survival(agent):
            if len(self.agents) == 1:
                logger.info("Combined input too low, no agent survived.")
            elif self._agent_selector.is_first():
                self.agents.remove(agent)
                self._agent_selector.reinit(self.agents)
            else:
                # remove agent and reconstruct agent selector
                previous_agent = agent
                next_agent = self._agent_selector.next()
                while next_agent != agent:
                    previous_agent = next_agent
                    next_agent = self._agent_selector.next()
                self.agents.remove(agent)
                self._agent_selector.reinit(self.agents)
                next_agent = self._agent_selector.next()
                while next_agent != previous_agent:
                    next_agent = self._agent_selector.next()

    def step(self, action):
        assert action.agent_id[0] == self.agent_selection

        if self.phase == self.LOCATION:
            assert action.location == self.LOCATION
            assert type(action) == PersonaActionChoice
            self._step_pre_round(action)
        elif self.phase == "office_after_round":
            assert action.location == self.LOCATION
            self._step_office_after_round(action)
        elif self.phase == "home":
            assert action.location == "home"
            self._step_home(action)
            if self._agent_selector.is_last():
                self.save_log()
                self.num_round += 1
                self._apply_perturbations()
                self.phase = self._phase_selector.next()
                self.terminations = self._get_terminations()

            self.agent_selection = self._agent_selector.next()

        return (
            self.agent_selection,
            self._observe(self.agent_selection),
            self.rewards,
            self.terminations,
        )

    def reset(self, seed: int = None) -> tuple[str, ActionObs]:
        self.random = np.random.RandomState(seed)

        self.agents = self.possible_agents[: self.cfg.num_agents]

        self.num_round = 0
        self.df_acc = []

        self.rewards = {}
        self.terminations = {}

        self.internal_global_state = {
            "num_agents": float(self.cfg.num_agents),
            "collected_resource": {},
            "chosen_action": {},
            "last_collected_resource": {},
            "next_location": {},
            "next_time": {},
            "action": {},
        }
        for agent in self.agents:
            self._init_agent(agent)

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._phase_selector = AgentSelector(
            [self.LOCATION, "office_after_round", "home"]
        )
        self.phase = self._phase_selector.next()

        self._apply_perturbations()

        return self.agent_selection, self._observe(self.agent_selection)

    def _get_terminations(self) -> dict[str, bool]:
        return (
            {
                "min_agents": len(self.agents) < 2,
                "max_rounds": self.num_round >= self.cfg.max_num_rounds
            }
        )
    
    def save_log(self):
        pd.concat(self.df_acc).to_json(
            f"{self.experiment_storage}/log_env.json", orient="records"
        )

    @abstractmethod
    def _prompt_round_payoffs(self, agent: str) -> str:
        pass
    
    @abstractmethod
    def _prompt_opponent_action_report(self) -> str:
        pass
    
    @abstractmethod
    def _agent_no_survival(self, agent: str) -> bool:
        pass

    @abstractmethod
    def _apply_perturbations(self) -> None:
        pass
    
    @abstractmethod
    def _generate_inputs_stochastic(self) -> None:
        pass
    
    @abstractmethod
    def _assign_resource(self) -> None:
        pass
