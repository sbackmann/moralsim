from moralsim.persona.cognition import StoreComponent
from moralsim.persona.common import PersonaIdentity
from moralsim.persona.embedding_model import EmbeddingModel
from moralsim.persona.memory.associative_memory import (
    Action,
    AssociativeMemory,
    Chat,
    Event,
    Thought,
)
from moralsim.utils import ModelWandbWrapper

from pathfinder import assistant, user
from typing import Callable


class MoralityStoreComponent(StoreComponent):

    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
        associative_memory: AssociativeMemory,
        embedding_model: EmbeddingModel,
        cfg,
        system_prompt_fn: Callable[[PersonaIdentity], str],
    ) -> None:
        super().__init__(
            model, model_framework, associative_memory, embedding_model, cfg
        )
        self.get_system_prompt = system_prompt_fn

    def prompt_importance_event(
        self, model: ModelWandbWrapper, persona: PersonaIdentity, event: Event
    ):
        lm = model.start_chain(persona.name, "cognition_perceive", "relevancy_event")

        with user():
            lm += f"{self.get_system_prompt(persona)}\n"
            lm += (
                "Task: Rate the significance of an event\nOn a scale of 1 to 10, where 1"
                " represents everyday, mundane activities (e.g., brushing teeth, making"
                " the bed) and 10 signifies events of extreme emotional significance"
                " (e.g., a romantic breakup, receiving a college acceptance letter),"
                f" evaluate the following event from {persona.name}'s perspective."
            )
            lm += f"\nEvent to rate: {event.description}\n"

        with assistant():
            lm += "Rating (1 to 10): "
            lm = model.select(
                lm,
                options=[str(i) for i in range(1, 11)],
                name="significance",
                default_value="5",  # assuming a neutral default value for better user guidance
            )
            importance_score = int(lm["significance"])

        model.end_chain(persona.name, lm)
        return importance_score


    def prompt_importance_thought(
        self, model: ModelWandbWrapper, persona: PersonaIdentity, thought: Thought
    ):
        lm = model.start_chain(
            persona.name, "cognition_retrieve", "prompt_importance_thought"
        )

        with user():
            lm += f"{self.get_system_prompt(persona)}\n"
            lm += (
                "Task: Rate the significance of a thought\nOn a scale from 1 to 10,"
                " where 1 indicates routine, everyday thoughts (e.g., needing to do"
                " chores) and 10 signifies thoughts of great importance (e.g., career"
                " aspirations, profound emotions), evaluate the following thought from"
                f" {persona.name}'s perspective."
            )
            lm += f"\nThought to rate:\n{thought.description}\n\n"

        with assistant():
            lm += "Rating (1 to 10): "
            lm = model.select(
                lm,
                options=[str(i) for i in range(1, 11)],
                default_value="5",  # Assuming a neutral default value for guidance
                name="significance_rating",
            )
            significance_rating = int(lm["significance_rating"])

        model.end_chain(persona.name, lm)
        return significance_rating


    def prompt_importance_action(
        self, model: ModelWandbWrapper, persona: PersonaIdentity, action: Action
    ):
        lm = model.start_chain(
            persona.name, "cognition_retrieve", "prompt_importance_action"
        )

        with user():
            lm += f"{self.get_system_prompt(persona)}\n"
            lm += (
                "Task: Rate the significance of an action\nOn a scale from 1 to 10,"
                " where 1 denotes routine, everyday actions (e.g., household chores) and"
                " 10 indicates actions of great importance or impact (e.g., career"
                " decisions, expressions of deep emotions), rate the significance of the"
                f" following action for {persona.name}."
            )
            lm += f"\nAction to rate:\n{action.description}\n\n"

        with assistant():
            lm += "Rating (1 to 10): "
            lm = model.select(
                lm,
                options=[str(i) for i in range(1, 11)],
                default_value="5",  # Setting a neutral default value for guidance
                name="significance_rating",
            )
            significance_rating = int(lm["significance_rating"])

        model.end_chain(persona.name, lm)
        return significance_rating


    def prompt_text_to_triple(model: ModelWandbWrapper, text: str):
        lm = model.start_chain("framework", "cognition_retrieve", "prompt_text_to_triple")

        with user():
            lm += f"Split the phrase into subject, predicate and object: {text}\n"

        with assistant():
            lm += "Subject: "
            lm = model.gen(lm, name="subject", stop_regex=r"\n")
            lm += "\nPredicate: "
            lm = model.gen(lm, name="predicate", stop_regex=r"\n")
            lm += "\nObject: "
            lm = model.gen(lm, name="object", stop_regex=r"\n")

        model.end_chain("framework", lm)
        return lm["subject"], lm["predicate"], lm["object"]
