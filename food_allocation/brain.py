# Brainクラス

import random
import numpy as np

from status import StockRemaining, StockChange, Satisfaction

GAMMA = 0.9
ALPHA = 0.005

EPSILON = 0.9


class Brain:
    def __init__(self, num_foods):
        shape = []
        # 状態数
        for i in range(num_foods):
            shape.append(len(StockRemaining))
        for i in range(num_foods):
            shape.append(len(StockChange))
        for i in range(num_foods):
            shape.append(len(Satisfaction))

        # 行動数
        shape.append(num_foods + 1)

        # self.Q = np.full(shape, -1000)
        self.Q = np.zeros(shape)
        print(f"Q shape: {shape} 要素数: {self.Q.size}")

    def update_Q(self, state, action, reward, state_next):

        if state_next is None:
            target = reward
        else:

            # sorted_Q = np.sort(self.Q[state_next])[::-1]
            # # print(sorted_Q)
            # for value in sorted_Q:
            #     if(value != 0):
            #         break
            max_Q = np.amax(self.Q[state_next])
            # print(max_Q)
            target = reward + GAMMA * max_Q

        predict = self.Q[state][action]

        self.Q[state][action] += ALPHA * (target - predict)

        # print(self.Q[state])

    # 選択肢の中から行動を決定
    def get_action(self, state, options, greedy):
        # state = self.convert_state(enum_state)

        # 何もしないという選択肢しかない場合
        # if(len(options) == 1):
        #     return options[0]

        if greedy is False:
            # ε-greedyで行動を決定する
            if np.random.rand() >= EPSILON:
                greedy = True

        if greedy:
            # Greedyに行動選択
            # Q値が大きい方から順に0でなく、かつ選択可能な行動を探す
            # print(self.Q[state])

            # 選択可能な行動のQ値を取り出す
            options_Q = self.Q[state][options]

            # 全て初期値だった場合はランダムに選択
            if np.all((options_Q == options_Q[0])):
                # print("全て初期値")
                action = random.choice(options)
            else:
                # arg_sorted_Q = np.argsort(self.Q[state])[::-1]
                # print(self.Q[state])
                max_arg = np.argmax(options_Q)
                action = options[max_arg]

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
