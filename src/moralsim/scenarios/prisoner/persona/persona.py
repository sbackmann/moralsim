import logging

from moralsim.persona import (
    PerceiveComponent,
    RetrieveComponent,
)
from moralsim.persona.common import (
    PersonaAction,
    PersonaActionChoice,
)
from moralsim.persona.embedding_model import EmbeddingModel
from moralsim.persona.memory import AssociativeMemory
from moralsim.scenarios.common.environment import ActionObs
from moralsim.scenarios.common.persona import MoralityDummyPersona, MoralityPersona
from moralsim.scenarios.common.persona.cognition import (
    MoralityPlanComponent,
    MoralityReflectComponent,
    MoralityStoreComponent
)
from moralsim.utils import ModelWandbWrapper

from .cognition import (
    PrisonerDummyActComponent,
    PrisonerActComponent,
)
from .cognition.utils import (
    get_decision_prompt_production,
    get_decision_prompt_base,
    get_decision_prompt_privacy,
    get_decision_prompt_venture,
    get_system_prompt_production,
    get_system_prompt_base,
    get_system_prompt_privacy,
    get_system_prompt_venture,
)

logger = logging.getLogger(__name__)


class PrisonerDummyPersona(MoralityDummyPersona):
    last_collected_resource_num: int

    def __init__(
        self,
        cfg,
        model: ModelWandbWrapper,
        actions: str | list,
        max_num_rounds: int
    ):
        super().__init__(cfg, model)
        self.act = PrisonerDummyActComponent(model, cfg, actions, max_num_rounds)
        self.act.init_persona_ref(self)


class PrisonerPersona(MoralityPersona):
    last_collected_resource_num: int
    other_personas: dict[str, "PrisonerPersona"]

    act: PrisonerActComponent

    def __init__(
        self,
        cfg,
        model: ModelWandbWrapper,
        framework_model: ModelWandbWrapper,
        embedding_model: EmbeddingModel,
        base_path: str,
        scenario: str,
        memory_cls: type[AssociativeMemory] = AssociativeMemory,
        perceive_cls: type[PerceiveComponent] = PerceiveComponent,
        retrieve_cls: type[RetrieveComponent] = RetrieveComponent,
        store_cls: type[MoralityStoreComponent] = MoralityStoreComponent,
        reflect_cls: type[MoralityReflectComponent] = MoralityReflectComponent,
        plan_cls: type[MoralityPlanComponent] = MoralityPlanComponent,
        act_cls: type[PrisonerActComponent] = PrisonerActComponent,
    ) -> None:
        if scenario == "pd_base":
            system_prompt_fn = get_system_prompt_base
            decision_prompt_fn = get_decision_prompt_base
            self.reflect_focus = "payoff"
        elif scenario == "pd_privacy":
            system_prompt_fn = get_system_prompt_privacy
            decision_prompt_fn = get_decision_prompt_privacy
            self.reflect_focus = "users"
        elif scenario == "pd_production":
            system_prompt_fn = get_system_prompt_production
            decision_prompt_fn = get_decision_prompt_production
            self.reflect_focus = "customers"
        elif scenario == "pd_venture":
            system_prompt_fn = get_system_prompt_venture
            decision_prompt_fn = get_decision_prompt_venture
            self.reflect_focus = "earnings"
        else:
            raise ValueError(f"Unexpected scenario name: {scenario}.")
        super().__init__(
            cfg,
            model,
            framework_model,
            embedding_model,
            base_path,
            system_prompt_fn,
            decision_prompt_fn,
            memory_cls,
            perceive_cls,
            retrieve_cls,
            store_cls,
            reflect_cls,
            plan_cls,
            act_cls,
        )
        print(self.cfg)

    def loop(self, obs: ActionObs) -> PersonaAction:
        res = []
        self.current_time = obs.current_time 

        self.perceive.perceive(obs)
        if obs.current_location == "office" and obs.phase == "office":
            retrieved_memory = self.retrieve.retrieve([obs.current_location], 10)
            num_resource, html_interactions = self.act.choose_action(
                retrieved_memory,
                obs.current_location,
                obs.current_time,
                str(obs.agent_resource_num),
            )
            action = PersonaActionChoice(
                (self.agent_id, "agent"),
                "office",
                num_resource,
                stats={
                    f"{self.agent_id}_input_value": obs.agent_resource_num,
                    f"{self.agent_id}_chosen_action": num_resource,
                },
                html_interactions=html_interactions,
            )
        elif obs.current_location == "office" and obs.phase == "office_after_round":
            # dummy action to register observation
            action = PersonaAction((self.agent_id, "agent"), "office")
        elif obs.current_location == "home":
            self.reflect.run([self.reflect_focus], self.cfg.reflect.exclude_thoughts_from_prompt)
            action = PersonaAction((self.agent_id, "agent"), "home")

        self.memory.save()
        return action
