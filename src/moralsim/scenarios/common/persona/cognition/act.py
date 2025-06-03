from datetime import datetime
from typing import Callable

from pathfinder import assistant, user
from moralsim.persona.cognition.act import ActComponent
from moralsim.persona.common import PersonaIdentity
from moralsim.utils import ModelWandbWrapper

from .utils import (
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)

class MoralityActComponent(ActComponent):
    def __init__(
        self, model: ModelWandbWrapper, model_framework: ModelWandbWrapper, cfg, system_prompt_fn: Callable[[PersonaIdentity], str], decision_prompt_fn: Callable[[str], str]
    ):
        super().__init__(model, model_framework, cfg)
        self.get_system_prompt: Callable = system_prompt_fn
        self.get_decision_prompt: Callable = decision_prompt_fn

    def choose_action(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
    ) -> float:
        print(self.persona.agent_id, context)
        res, html = self.prompt_choose_action(
            self.model,
            self.persona.identity,
            retrieved_memories,
            current_location,
            current_time,
            context,
        )
        res = float(res)
        return res, [html]

    def prompt_choose_action(
        self,
        model: ModelWandbWrapper,
        identity: PersonaIdentity,
        memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
    ) -> float:
        lm = model.start_chain(
            identity.name, "cognition_act", "choose_act_options"
        )

        with user():
            lm += self.get_system_prompt(identity)
            lm += "\n"
            lm += location_time_info(current_location, current_time)
            lm += memory_prompt(identity, memories)
            lm += "\n"
            lm += self.get_decision_prompt(context)
            lm += "\n"
            lm += reasoning_steps_prompt()
            lm += self.get_format_answer_prompt()

        with assistant():
            lm = model.gen(
                lm,
                "reasoning",
                stop_regex=r"Answer:|So, the answer is:|he final answer is|\*\*Answer\*\*:",
                save_stop_text=True,
                max_tokens=8000,
            )
            lm = model.find(
                lm,
                regex=r"\d*\.?\d+",
                default_value="0",
                # stop_regex=f"tons",
                name="option",
            )
            option = float(lm["option"])

        model.end_chain(identity.name, lm)

        return option, lm.html()
    
    def get_format_answer_prompt(self) -> str:
        return 'Put the final answer after "Answer:". Example: "Answer: ..."'
    
    def get_decision_prompt(self, context: str) -> str:
        raise NotImplementedError
