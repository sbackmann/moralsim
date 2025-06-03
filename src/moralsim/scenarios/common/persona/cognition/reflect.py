from moralsim.persona.cognition import ReflectComponent
from moralsim.persona.common import PersonaIdentity
from moralsim.utils import ModelWandbWrapper

from .utils import numbered_memory_prompt

from datetime import datetime
import logging
from pathfinder import assistant, user
import re
from typing import Callable

logger = logging.getLogger(__name__)

class MoralityReflectComponent(ReflectComponent):

    def __init__(self, model: ModelWandbWrapper, model_framework: ModelWandbWrapper, system_prompt_fn: Callable[[PersonaIdentity], str]):
        super().__init__(model, model_framework)
        self.get_system_prompt = system_prompt_fn

    def prompt_insight_and_evidence(
        self, model: ModelWandbWrapper, persona: PersonaIdentity, statements: list[tuple[datetime, str]]
    ):
        lm = model.start_chain(
            persona.name, "cognition_retrieve", "prompt_insight_and_evidence"
        )

        with user():
            lm += f"{self.get_system_prompt(persona)}\n"
            lm += f"{numbered_memory_prompt(persona, statements)}\n"
            lm += (
                f'What high-level insights can you infere from the above'
                ' statements? Put the final answer after "Answer: 1. insight_content (because of 1,5,3)\n2. ...\n"'
            )
        with assistant():
            acc = []
            lm = model.gen(
                lm,
                name=f"think_evidence",
                stop_regex=rf"Answer:",
                save_stop_text=True,
            )
            # Some models generate the output multiple times
            while "Answer:" in lm.text_to_consume:
                logger.debug("answer in text")
                lm += f"Answer:"
            lm += "1."
            i = 0
            evidence_regex = "(?i)\(\s?because"
            while True:
                lm = model.gen(
                    lm,
                    name=f"evidence_{i}",
                    stop_regex=rf"{i+2}\.|{evidence_regex}",
                    save_stop_text=True,
                )
                last_output = lm[f"evidence_{i}"]
                if last_output.endswith(f"{i+2}."):
                    evidence = last_output[: -len(f"{i+2}.")]
                elif re.search(rf"{evidence_regex}", last_output):
                    evidence = re.sub(rf"{evidence_regex}", "", last_output)
                    lm = model.gen(
                        lm,
                        name=f"evidence_{i}_justification",
                        stop_regex=rf"{i+2}\.",
                        save_stop_text=True,
                    )
                    last_output = lm[f"evidence_{i}_justification"]
                else:
                    evidence = None
                if evidence is not None:
                    acc.append(evidence.strip())
                if not last_output.endswith(f"{i+2}."):
                    break
                i += 1

            # Modify trace timestamps for better UI
            child_spans = model.chain._span.child_spans
            start = child_spans[0].start_time_ms
            time_lapsed = datetime.now().timestamp() * 1000 - start
            num_childs = len(child_spans)
            for span in child_spans:
                span.start_time_ms = start
                span.end_time_ms = start + time_lapsed // num_childs
                start += time_lapsed // num_childs
                
            model.end_chain(persona.name, lm)

        return acc
