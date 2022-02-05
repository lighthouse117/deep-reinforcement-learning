# Agentクラス

import numpy as np
from brain import Brain
from status import StockRemaining, StockChange, Satisfaction, Progress
from config import EnvironmentSettings as es


class Agent:
    def __init__(self, name, request, f):
        self.name = name
        self.REQUEST = request
        self.brain = Brain(f)
        self.f = f

    def reset(self, env_stock, greedy):
        self.current_requests = self.REQUEST.copy()
        self.stock = np.zeros(es.NUM_FOODS, dtype=np.int64)

        self.food_done = False
        self.learning_done = False
        self.old_env_stock = env_stock.copy()
        if greedy:
            self.brain.get_TD_average()

    def learn(self, state, action, reward, state_next, alpha, greedy):
        # print("state:")
        # self.print_state(state)
        # if state_next is not None:
        #     print("state_next: ")
        #     self.print_state(state_next)
        # print(f"action: {action}")
        # print(f"reward: {reward}")
        if state == state_next:
            if greedy:
                print(f"{self.name}: 状態が変化していません", file=self.f)
        else:
            self.brain.update_Q(state, action, reward,
                                state_next, alpha, greedy)
        # print(f"{self.name} Q値を更新")

    # 行動（どの食品を取得するか）を決定
    def decide_action(self, state, env_stock, greedy, epsilon):

        # if greedy:
        #     self.print_state(state)

        # 要求が満たされた後も自由に行動して学習をさせる
        # （十分に獲得したのにさらに手を出そうとした場合、罰を与える）

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

        action_options.append(es.NUM_FOODS)

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

    def observe_state(self, env_stock, episode_terminal):
        remainings = []
        changes = []
        satisfactions = []
        progress = []

        granularity = len(StockRemaining) - 2
        for amount, original in zip(env_stock, es.FOODS):
            section = round(original / granularity)
            if amount == 0:
                remainings.append(StockRemaining.NONE)
            elif amount < section:
                remainings.append(StockRemaining.FEW)
            elif amount < original:
                remainings.append(StockRemaining.MANY)
            else:
                remainings.append(StockRemaining.FULL)

        difference = self.old_env_stock - env_stock
        for diff in difference:
            if diff == 0:
                changes.append(StockChange.NONE)
            elif diff == 1:
                changes.append(StockChange.SLIGHTLY)
            elif diff == 2:
                changes.append(StockChange.SOMEWHAT)
            else:
                changes.append(StockChange.GREATLY)

        satisfaction_rates = self.stock / self.REQUEST
        for rate in satisfaction_rates:
            # print(f"satisfaction = {diff}")
            if rate < 0.5:
                satisfactions.append(Satisfaction.HARDLY)
            elif rate < 1:
                satisfactions.append(Satisfaction.SOMEWHAT)
            elif rate == 1:
                satisfactions.append(Satisfaction.COMLETELY)
            else:
                satisfactions.append(Satisfaction.OVERLY)

        if episode_terminal:
            progress.append(Progress.DONE)
        else:
            progress.append(Progress.ONGOING)

        state = tuple(remainings + changes + satisfactions + progress)

        # state = tuple(remainings + satisfactions + progress)

        self.old_env_stock = env_stock.copy()

        return state

    # def get_state(self, env_state):
    #     personal_state = []

    #     # section = 1 / (len(Satisfaction) - 1)
    #     satisfactions = self.stock / self.REQUESTS

    #     for satisfaction in satisfactions:
    #         # print(f"satisfaction = {diff}")
    #         if satisfaction < 0.5:
    #             personal_state.append(Satisfaction.HARDLY)
    #         elif satisfaction < 1:
    #             personal_state.append(Satisfaction.SOMEWHAT)
    #         elif satisfaction == 1:
    #             personal_state.append(Satisfaction.COMLETELY)
    #         else:
    #             personal_state.append(Satisfaction.OVERLY)

    #     state = env_state + tuple(personal_state)
    #     # print(f"{self.name} state: {state}")

    #     return state

    def grab_food(self, food):
        # 手元の在庫が1つ増える
        self.stock[food] += 1
        # 要求リストから1つ減らす
        self.current_requests[food] -= 1

        self.check_satisfied()

    def check_satisfied(self):
        if np.all(self.current_requests == 0):
            # 要求がすべて満たされたことを記録
            self.done = True
            # print(f"{self.name} 要求がすべて満たされました")

        # TODO: 食品ごとに調べる（要求があっても食品の在庫がない場合）

    def get_satisfaction(self):
        diffs = self.REQUEST - self.stock
        diff_rates = diffs / self.REQUEST * 10
        abs_diffs = np.absolute(diff_rates)
        satisfaction = - np.sum(abs_diffs)

        self.satisfaction = satisfaction
        return satisfaction

    # def get_reward(self):

        # def get_loss_value(self):
        #     diffs = self.REQUESTS - self.stock
        #     # print(f"diffs: {diffs}")
        #     diff_rates = diffs / self.REQUESTS * 100
        #     # print(f"diff_rates: {diff_rates}")
        #     weighted_diffs = np.where(
        #         diff_rates < 0, np.absolute(diff_rates * 10), diff_rates)
        #     # print(f"weighted_diffs: {weighted_diffs}")

        #     loss = np.average(weighted_diffs)
        #     return loss

    def print_state(self, state):
        num = es.NUM_FOODS
        print(f"{self.name} State: ", end="", file=self.f)

        print("Remaining[ ", end="", file=self.f)
        for i in range(num):
            print(f"{state[i].name} ", end="", file=self.f)
        print("], Change[ ", end="", file=self.f)
        for i in range(num, num * 2):
            print(f"{state[i].name} ", end="", file=self.f)
        print("], Satisfaction[ ", end="", file=self.f)
        for i in range(num * 2, num * 3):
            print(f"{state[i].name} ", end="", file=self.f)

        print(f"] Progress[{state[num * 3].name}]", file=self.f)

        # print("Remaining[ ", end="", file=self.f)
        # for i in range(num):
        #     print(f"{state[i].name} ", end="", file=self.f)
        # print("], Satisfaction[ ", end="", file=self.f)
        # for i in range(num, num * 2):
        #     print(f"{state[i].name} ", end="", file=self.f)

        # print(f"] Progress[{state[num * 2].name}]", file=self.f)
