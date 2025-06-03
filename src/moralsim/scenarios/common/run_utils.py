import os
import sys
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf

from moralsim.persona import EmbeddingModel, PersonaAgent
from moralsim.persona.common import PersonaIdentity
from moralsim.utils import ModelWandbWrapper
from moralsim.scenarios.common import ActionObs
from .environment import MoralityPerturbationEnv
from .persona import MoralityDummyPersona

def init_all_personas(
    personas: dict[str, MoralityDummyPersona | PersonaAgent],
    num_personas: int,
    cfg
) -> tuple[dict[str, str], dict[str, str]]:
    identities = {}
    for i in range(num_personas):
        persona_id = f"persona_{i}"
        identities[persona_id] = PersonaIdentity(
            agent_id=persona_id, **cfg.personas[persona_id]
        )

    # Standard setup
    agent_name_to_id = {obj.name: k for k, obj in identities.items()}
    agent_name_to_id["framework"] = "framework"
    agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

    for persona in personas:
        personas[persona].init_persona(persona, identities[persona], social_graph=None)

    for persona in personas:
        for other_persona in personas:
            # also add self reference, for conversation
            personas[persona].add_reference_to_other_persona(personas[other_persona])
    return agent_name_to_id, agent_id_to_name

def run_step(
    env: MoralityPerturbationEnv,
    agent: MoralityDummyPersona | PersonaAgent,
    obs: ActionObs,
    num_personas: int,
    wandb_logger: ModelWandbWrapper,
) -> tuple[bool, str, ActionObs]:
    has_next_step = True
    action = agent.loop(obs)

    (
        agent_id,
        obs,
        rewards,
        termination,
    ) = env.step(action)

    stats = {}
    STATS_KEYS = [
        "conversation_resource_limit",
        *[f"persona_{i}_input_value" for i in range(num_personas)],
        *[f"persona_{i}_chosen_action" for i in range(num_personas)],
    ]
    for s in STATS_KEYS:
        if s in action.stats:
            stats[s] = action.stats[s]

    if np.any(list(termination.values())):
        wandb_logger.log_game(
            {
                **stats,
            },
            last_log=True,
        )
        has_next_step = False
    else:
        wandb_logger.log_game(
            {
                **stats,
            }
        )
    return has_next_step, agent_id, obs
