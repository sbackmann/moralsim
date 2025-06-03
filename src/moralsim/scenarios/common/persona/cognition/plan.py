from pathfinder import assistant, system, user
from moralsim.persona.cognition.plan import PlanComponent
from moralsim.utils import ModelWandbWrapper


class MoralityPlanComponent(PlanComponent):
    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
    ):
        super().__init__(model, model_framework)
