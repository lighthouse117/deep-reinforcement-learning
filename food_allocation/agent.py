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

    def learn(self, state, action, reward, state_next, alpha):
        # print("state:")
        # self.print_state(state)
        # if state_next is not None:
        #     print("state_next: ")
        #     self.print_state(state_next)
        # print(f"action: {action}")
        # print(f"reward: {reward}")
        self.brain.update_Q(state, action, reward, state_next, alpha)
        # print(f"{self.name} Q値を更新")

    # 行動（どの食品を取得するか）を決定
    def decide_action(self, state, env_stock, greedy, epsilon):

        # if greedy:
        #     self.print_state(state)

        # 要求が満たされた後も自由に行動して学習をさせる
        # （十分に獲得したのにさらに手を出そうとした場合、罰を与える）

        if np.all(self.current_requests == 0):
            # 要求がすべて満たされたことを記録
            self.done = True
            # print(f"{self.name} 要求がすべて満たされました")

        # 行動の候補
        action_options = []

        # 本部に在庫がある食品を候補に入れる
        for food in range(len(env_stock)):
            if env_stock[food] != 0:
                action_options.append(food)

        # 自分が要求していた食品（既に満たされたかは関係ない） & 本部に在庫がある食品を候補に入れる
        # for food, amount in enumerate(self.REQUESTS):
        #     if(amount != 0 and env_stock[food] != 0):
        #         action_options.append(food)

        # 在庫切れ
        # if len(action_options) == 0:
            # 候補の食品の在庫がすべてなかった場合「何もしない」
            # action = len(self.stock)
            # self.done = True
            # print(f"{self.name} ほしい食品の在庫がありません")
            # return action

        # 「何もしない」という選択肢も候補に加える
        action_options.append(len(self.REQUESTS))

        # if greedy:
        #     print(self.name + "  ", end="")

        # 行動を決定
        action = self.brain.get_action(state, action_options, greedy, epsilon)

        # if greedy:
        #     if action == self.NUM_FOODS:
        #         print(" 行動: 何もしない", end="")
        #         pass
        #     else:
        #         print(f" 行動: 食品{action}を取る", end="")
        #         pass

        return action

    def get_state(self, env_state):
        personal_state = []

        # section = 1 / (len(Satisfaction) - 1)
        satisfactions = self.stock / self.REQUESTS

        for satisfaction in satisfactions:
            # print(f"satisfaction = {diff}")
            if satisfaction < 0.5:
                personal_state.append(Satisfaction.HARDLY)
            elif satisfaction < 1:
                personal_state.append(Satisfaction.SOMEWHAT)
            elif satisfaction == 1:
                personal_state.append(Satisfaction.COMLETELY)
            else:
                personal_state.append(Satisfaction.OVERLY)

        state = env_state + tuple(personal_state)
        # print(f"{self.name} state: {state}")

        return state

    def get_food(self, food):
        # 手元の在庫が1つ増える
        self.stock[food] += 1
        # 要求リストから1つ減らす
        self.current_requests[food] -= 1

    def get_loss_value(self):
        diffs = self.REQUESTS - self.stock
        # print(f"diffs: {diffs}")
        diff_rates = diffs / self.REQUESTS * 100
        # print(f"diff_rates: {diff_rates}")
        weighted_diffs = np.where(
            diff_rates < 0, np.absolute(diff_rates * 10), diff_rates)
        # print(f"weighted_diffs: {weighted_diffs}")

        loss = np.average(weighted_diffs)
        return loss

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
