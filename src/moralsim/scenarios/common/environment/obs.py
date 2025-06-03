from datetime import datetime

from moralsim.persona.common import ChatObservation, PersonaOberservation


class ActionObs(PersonaOberservation):

    agent_resource_num: dict[str, int]

    def __init__(
        self,
        phase: str,
        current_location: str,
        current_location_agents: dict[str, str],
        current_time: datetime,
        events: list,
        context: str,
        agent_resource_num: dict[str, int],
    ) -> None:
        super().__init__(
            phase,
            current_location,
            current_location_agents,
            current_time,
            events,
            context,
        )
        self.agent_resource_num = agent_resource_num
