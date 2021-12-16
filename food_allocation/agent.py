# Agentクラス

import numpy as np
from brain import Brain
from status import Satisfaction


class Agent:
    def __init__(self, name, max_units, num_foods, requests):
        # num_states = pow(max_units, 2 * num_foods)
        # num_actions = num_foods + 1
        self.name = name
        self.NUM_FOODS = num_foods
        self.brain = Brain(num_foods)
        self.REQUESTS = np.array(requests)

    def reset(self, env_state):
        self.current_requests = self.REQUESTS.copy()
        self.stock = np.zeros(self.NUM_FOODS, dtype=np.int64)
        self.done = False
        state = self.get_state(env_state)
        return state

    def learn(self, state, action, reward, state_next):
        self.brain.update_Q(state, action, reward, state_next)
        # print(f"{self.name} Q値を更新")

    def decide_action(self, state, env_stock, greedy):
        if self.done:
            # 既にとれる行動がもうないので「何もしない」
            action = len(self.stock)
            # print(f"{self.name} とれる行動がありません")
            return action

        if np.all(self.current_requests == 0):
            # 要求がすべて満たされてる場合は「何もしない」
            action = len(self.stock)
            self.done = True
            # print(f"{self.name} 要求がすべて満たされました")
            return action

        # 行動の候補
        action_options = []

        # 要求個数が0でない & 本部に在庫がある食品を候補に入れる
        for food, amount in enumerate(self.current_requests):
            if(amount != 0 and env_stock[food] != 0):
                action_options.append(food)

        # 在庫切れ
        if len(action_options) == 0:
            # 候補の食品の在庫がすべてなかった場合「何もしない」
            action = len(self.stock)
            self.done = True
            # print(f"{self.name} ほしい食品の在庫がありません")
            return action

        # 「何もしない」という選択肢も候補に加える
        action_options.append(len(self.REQUESTS))

        # 行動を決定
        action = self.brain.get_action(state, action_options, greedy)

        return action

    def get_state(self, env_state):
        personal_state = []

        difference = self.stock / self.REQUESTS
        for diff in difference:
            # print(f"satisfaction = {diff}")
            if diff < 0.5:
                personal_state.append(Satisfaction.NOT)
            elif diff < 1.0:
                personal_state.append(Satisfaction.SOMEWHAT)
            else:
                personal_state.append(Satisfaction.PERFECT)

        state = env_state + tuple(personal_state)
        # print(f"{self.name} state: {state}")
        # self.print_state(state)
        return state

    def get_food(self, food):
        # 手元の在庫が1つ増える
        self.stock[food] += 1
        # 要求リストから1つ減らす
        self.current_requests[food] -= 1

    def get_satisfaction(self):
        rates = []
        for stock, request in zip(self.stock, self.REQUESTS):
            rates.append(stock / request * 100)
        satisfaction = np.average(np.array(rates))
        # print(f"\n------ {self.name} ------")
        # print(f"要求: {self.REQUESTS}")
        # print(f"獲得: {self.stock}")
        # print(f"満足度: {satisfaction:.1f}%")
        return satisfaction

    def print_state(self, state):
        num = self.NUM_FOODS
        print(f"{self.name} State: ", end="")

        print("Remaining[ ", end="")
        for i in range(num):
            print(f"{state[i].name} ", end="")
        print("], Change[ ", end="")
        for i in range(num, num * 2):
            print(f"{state[i].name} ", end="")
        print("], Satisfaction[ ", end="")
        for i in range(num * 2, num * 3):
            print(f"{state[i].name} ", end="")
        print("]")
