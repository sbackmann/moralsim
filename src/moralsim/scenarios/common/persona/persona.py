import logging
import os
from typing import Callable

from moralsim.persona import PersonaAgent

from moralsim.persona.common import (
    PersonaAction,
    PersonaActionChoice,
    PersonaIdentity,
)
from moralsim.scenarios.common.environment import ActionObs
from moralsim.persona.embedding_model import EmbeddingModel
from moralsim.persona.memory import AssociativeMemory, Scratch
from moralsim.persona.cognition import PerceiveComponent, RetrieveComponent
from moralsim.utils import ModelWandbWrapper

from .cognition import (
    MoralityActComponent,
    MoralityPlanComponent,
    MoralityReflectComponent,
    MoralityStoreComponent,
)

logger = logging.getLogger(__name__)


class MoralityDummyPersona:
    last_collected_resource_num: int

    def __init__(
        self,
        cfg,
        model: ModelWandbWrapper,
    ):
        self.model_wrapper = model
        self.cfg = cfg
        self.other_personas: dict[str, MoralityPersona | MoralityDummyPersona] = {}
        self.other_personas_from_id: dict[str, MoralityPersona | MoralityDummyPersona] = {}
        self.act: MoralityActComponent = None

    def init_persona(self, agent_id: int, identity: PersonaIdentity, social_graph):
        self.agent_id = agent_id
        self.identity = identity

    def add_reference_to_other_persona(self, persona: "MoralityPersona | MoralityDummyPersona"):
        self.other_personas[persona.identity.name] = persona
        self.other_personas_from_id[persona.agent_id] = persona
        self.act.add_reference_to_other_persona(persona)

    def loop(self, obs: ActionObs):
        self.current_time = obs.current_time  # update current time

        # phase based game
        if obs.current_location == "office" and obs.phase == "office":
            # Stage 1. Earnings situation / Stage 2. Buiness partner’s decisions
            num_resource = self.act.choose_action(
                obs.agent_resource_num,
            )
            action = PersonaActionChoice(
                (self.agent_id, "dummy"),
                "office",
                num_resource,
                stats={
                    f"{self.agent_id}_input_value": obs.agent_resource_num,
                    f"{self.agent_id}_chosen_action": num_resource,
                },
                html_interactions=str(num_resource),
            )
        else:
            # dummy action to register observation
            action = PersonaAction((self.agent_id, "dummy"), obs.current_location)

        return action

class MoralityPersona(PersonaAgent):
    def __init__(
        self,
        cfg,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
        embedding_model: EmbeddingModel,
        base_path: str,
        system_prompt_fn: Callable[[PersonaIdentity], str],
        decision_prompt_fn: Callable[[str], str],
        memory_cls: type[AssociativeMemory] = AssociativeMemory,
        perceive_cls: type[PerceiveComponent] = PerceiveComponent,
        retrieve_cls: type[RetrieveComponent] = RetrieveComponent,
        store_cls: type[MoralityStoreComponent] = MoralityStoreComponent,
        reflect_cls: type[MoralityReflectComponent] = MoralityReflectComponent,
        plan_cls: type[MoralityPlanComponent] = MoralityPlanComponent,
        act_cls: type[MoralityActComponent] = MoralityActComponent,
    ) -> None:
        self.cfg = cfg
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

        self.memory = memory_cls(base_path)
        self.perceive = perceive_cls(model, model_framework)
        self.retrieve = retrieve_cls(
            model, model_framework, self.memory, embedding_model
        )
        self.store = store_cls(
            model, model_framework, self.memory, embedding_model, self.cfg.store, system_prompt_fn
        )
        self.reflect = reflect_cls(model, model_framework, system_prompt_fn)
        self.plan = plan_cls(model, model_framework)
        self.act = act_cls(
            model,
            model_framework,
            self.cfg.act,
            system_prompt_fn,
            decision_prompt_fn,
        )

        self.perceive.init_persona_ref(self)
        self.retrieve.init_persona_ref(self)
        self.store.init_persona_ref(self)
        self.reflect.init_persona_ref(self)
        self.plan.init_persona_ref(self)
        self.act.init_persona_ref(self)

        self.other_personas: dict[str, MoralityPersona | MoralityDummyPersona] = {}
        self.other_personas_from_id: dict[str, MoralityPersona | MoralityDummyPersona] = {}

    def add_reference_to_other_persona(self, persona: "MoralityPersona | MoralityDummyPersona"):
        self.other_personas[persona.identity.name] = persona
        self.other_personas_from_id[persona.agent_id] = persona
        self.perceive.add_reference_to_other_persona(persona)
        self.retrieve.add_reference_to_other_persona(persona)
        self.store.add_reference_to_other_persona(persona)
        self.reflect.add_reference_to_other_persona(persona)
        self.plan.add_reference_to_other_persona(persona)
        self.act.add_reference_to_other_persona(persona)
