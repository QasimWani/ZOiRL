# Defining the CEM Agent
# A buffer will also be used to keep the last 3 elite set of zetas



class CEM_Agent(object):
    def __init__(self,building_ids,
        buildings_states_actions,
        building_info,
        observation_spaces,
        action_spaces,
        num_actions: int = 3,
        num_buildings: int = 9,
        rbc_threshold: int = 336,  # 2 weeks by default
        is_oracle: bool = True
        ):


        self.env = env
        self.building_ids = building_ids
        self.buildings_states_actions = buildings_states_actions
        self.building_info = building_info
        self.obsevation_spaces = observation_spaces
        self.action_spaces = action_spaces

        self.total_it = 0
        self.rbc_threshold = rbc_threshold

    # Returns actions to be perfromed given a present state

    def select_action(self, state):

        zeta = CEM_actions.get_zeta(state)
        actions = CEM_actions.get_actions(state, zeta)

        return actions
