import numpy as np
from agent import Agent
import random as rd

class ActionLogic:
    def __init__(self, refresh_rate=10, nb_rewards=3):
        self.refresh_rate = refresh_rate
        self.current_refresh = refresh_rate
        self.nb_rewards = nb_rewards
        self.pos_visited = set()

    def choose_action(self, agent: Agent) -> str:
        if self.current_refresh == 0:
            self.pos_visited.clear()
            self.current_refresh = self.refresh_rate
        self.current_refresh -= 1

        go_to_reward = 0
        if any(reward > 0 for reward in agent.known_rewards):
            go_to_reward = 0.014 * np.max(agent.known_rewards) + 0.05 * len(self.pos_visited)
            if self.nb_rewards == 1:
                go_to_reward = 1

        stay = go_to_reward if agent.known_rewards[agent.position] == np.max(agent.known_rewards) else 1 - go_to_reward

        action = rd.choices(["none", "move"], weights=[stay, 1 - stay])[0]

        if action == "none":
            return action

        left_weight = 0.5
        right_weight = 0.5
        if agent.position not in self.pos_visited:
            self.pos_visited.add(agent.position)

        visited_left = sum(1 for pos in self.pos_visited if pos < agent.position)
        visited_right = sum(1 for pos in self.pos_visited if pos > agent.position)
        left_weight = max(0, min(left_weight - visited_left * 0.3, 1))
        right_weight = max(0, min(right_weight + visited_right * 0.3, 1))

        distance_to_middle = abs(agent.position - 3.5)  # 3.5 is the middle position
        left_weight = max(0, min(left_weight - distance_to_middle * 0.08, 1))
        right_weight = max(0, min(right_weight + distance_to_middle * 0.08, 1))
        return rd.choices(["left", "right"], weights=[left_weight, right_weight])[0]


def my_new_policy(agent: Agent, refresh_rate=10, nb_rewards=3) -> str:
    global glob_action_instance

    if 'glob_action_instance' not in globals():
        glob_action_instance = ActionLogic(refresh_rate, nb_rewards)

    action = glob_action_instance.choose_action(agent)
    return action
