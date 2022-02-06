# Brainクラス

import random
import numpy as np

from status import StockRemaining, StockChange, Satisfaction, Progress
from config import LearningParameters as lp
from config import EnvironmentSettings as es


class Brain:
    def __init__(self, f):
        self.f = f
        self.TDs = []
        self.max_TD = 0.0
        shape = []

        # 状態数
        for i in range(es.NUM_FOODS):
            shape.append(len(StockRemaining))
        for i in range(es.NUM_FOODS):
            shape.append(len(StockChange))
        for i in range(es.NUM_FOODS):
            shape.append(len(Satisfaction))
        shape.append(len(Progress))

        # 行動数
        shape.append(es.NUM_FOODS + 1)

        # Qテーブルを初期化
        self.Q = np.full(shape, -1.0)
        # self.Q = np.zeros(shape)

        print(f"Qテーブルのshpae: {shape}   要素数: {self.Q.size:,}", file=self.f)

    def update_Q(self, state, action, reward, state_next, alpha, greedy):

        if state_next[-1] is Progress.DONE:
            target = reward
        else:
            # sorted_Q = np.sort(self.Q[state_next])[::-1]
            # # print(sorted_Q)
            # for value in sorted_Q:
            #     if(value != 0):
            #         break
            # max_Q = np.amax(self.Q[state_next])
            # target = reward + lp.GAMMA * max_Q
            next_Q = []
            for i in range(es.NUM_FOODS):
                if state_next[i] != StockRemaining.NONE:
                    next_Q.append(self.Q[state_next][i])
            next_Q.append(self.Q[state_next][-1])

            max_Q = max(next_Q)
            target = reward + lp.GAMMA * max_Q

        # if state_next[-1] is Progress.DONE:
        #     target = reward
        # else:
        #     next_Q = []
        #     for i in range(es.NUM_FOODS):
        #         if state_next[i] != StockRemaining.NONE:
        #             next_Q.append(self.Q[state_next][i])
        #     next_Q.append(self.Q[state_next][-1])

        #     max_Q = max(next_Q)
        #     target = reward + lp.GAMMA * max_Q

        current = self.Q[state][action]

        diff = target - current

        td = abs(diff)

        self.TDs.append(td)
        if td > self.max_TD:
            self.max_TD = td

        # if reward == -60:
        #     print(f"前: {self.Q[state][action]}")

        # if abs(diff) > 50:
        #     pass

        # print(f"before: {self.Q[state][action]}")

        self.Q[state][action] = self.Q[state][action] + alpha * diff

        # if reward == -60:
        #     print(f"後: {self.Q[state][action]}\n")

        # if self.Q[state][action] != 0:
        #     print(self.Q[state][action])

        # print(reward)

        # print(f"after: {self.Q[state][action]}\n")

        # print(self.Q[state])

    # 選択肢の中から行動を決定

    def get_action(self, state, options, greedy, epsilon):
        # state = self.convert_state(enum_state)

        # 何もしないという選択肢しかない場合
        # if(len(options) == 1):
        #     return options[0]
        # if greedy:
        #     print(self.Q[state], end="")

        if greedy is False:
            # ε-greedyで行動を決定する
            if np.random.rand() >= epsilon:
                greedy = True

        if greedy:
            # Greedyに行動選択
            # Q値が大きい方から順に0でなく、かつ選択可能な行動を探す

            # 選択可能な行動のQ値を取り出す
            options_Q = self.Q[state][options]

            max_arg = np.argmax(options_Q)
            action = options[max_arg]

            # 全て初期値だった場合はランダムに選択
            # if np.all((options_Q == options_Q[0])):
            #     # print("全て初期値")
            #     action = random.choice(options)
            # else:
            # arg_sorted_Q = np.argsort(self.Q[state])[::-1]
            # print(self.Q[state])

            # for food in arg_sorted_Q:
            #     # 選択肢に含まれているか
            #     if food in options:
            #         action = food
            #         break
            # if self.Q[state][food] != 0:
            # break
            # print(f"greedy action: {action}")
        else:
            # Randomに行動選択
            # 候補の中からランダムに選ぶ
            # print(f"action options: {options}")
            action = random.choice(options)
            # print(f"random action: {action}")

        return action

    def get_TD_average(self):
        if self.TDs:
            TD_ave = np.average(np.array(self.TDs))
            # print(self.TDs)
            print(
                f"TD誤差 {len(self.TDs)}回分の平均: {TD_ave:.5f}  最大値: {self.max_TD:.5f}", file=self.f)
            print(f"TD誤差の平均: {TD_ave:.5f}  最大値: {self.max_TD:.5f}")
            self.max_TD = 0
            self.TDs = []

    def Q_str(self, state):
        q = []
        for i in range(es.NUM_FOODS):
            if state[i] == StockRemaining.NONE:
                q.append(None)
            else:
                q.append(self.Q[state][i])
        q.append(self.Q[state][-1])

        return np.array2string(np.array(q))
