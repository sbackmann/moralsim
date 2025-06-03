from moralsim.persona.cognition.act import ActComponent
from moralsim.utils import ModelWandbWrapper


class PublicGoodsDummyActComponent(ActComponent):
    def __init__(self, model: ModelWandbWrapper, cfg, actions: str | list, max_num_rounds: int):
        super().__init__(model, None)
        self.actions = self.compute_action_trajectory(actions, max_num_rounds)
        self.current_round = 0


    def compute_action_trajectory(self, actions: str | list, max_num_rounds: int):
        """Determine fixed actions trajectory
            "staged": Start truthful (1/3 of the rounds), increasingly underreport, once survival is
        """
        if isinstance(actions, list):
            if len(actions) != max_num_rounds:
                raise ValueError("List of actions was passed which doesn't match number of rounds.")
            return actions
        elif actions == "cooperate":
            return [1] * max_num_rounds
        elif actions == "defect":
            return [0] * max_num_rounds
        elif actions == "worsening":
            len_stage = max_num_rounds // 3
            action_list = [1] * len_stage
            action_list += [0.5] * (len_stage // 2) 
            action_list += [0] * int(len_stage * 1.5)
            if (missing := max_num_rounds - len(action_list)) > 0:
                action_list += [0] * missing
            return action_list
        else:
            raise ValueError(f"Unexpected action strategy: {actions}")


    def choose_action(
        self,
        context: str,
    ):
        res = int(self.actions[self.current_round] * int(context))
        self.current_round += 1
        return res
