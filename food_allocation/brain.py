# Brainクラス

import random
import numpy as np

GAMMA = 0.98
ALPHA = 0.1

EPSILON = 0.9


class Brain:
    def __init__(self, max_units, num_foods):
        shape = []
        for i in range(num_foods * 2):
            shape.append(max_units + 1)
        for i in range(num_foods):
            shape.append(2 * max_units + 1)
        shape.append(num_foods + 1)

        self.Q = np.zeros((shape))
        print(f"Q shape: {shape} 要素数: {self.Q.size}")

    def update_Q(self, state, action, reward, state_next):
        if state_next is None:
            target = reward
        else:
            max_Q = np.amax(self.Q[state_next])
            target = reward + GAMMA * max_Q

        predict = self.Q[state][action]

        self.Q[state][action] += ALPHA * (target - predict)

    # 選択肢の中から行動を決定
    def get_action(self, state, options, greedy):
        # 何もしないという選択肢しかない場合
        if(len(options) == 1):
            return options[0]

        if greedy is False:
            # ε-greedyで行動を決定する
            if np.random.rand() >= EPSILON:
                greedy = True

        if greedy:
            # Greedy
            # Q値が大きい方から順にとれる行動を探す
            sorted_Q = np.argsort(self.Q[state])[::-1]
            for food in sorted_Q:
                # 選択肢に含まれているか
                if(food in options):
                    action = food
                    break
            # print(f"greedy action: {action}")
        else:
            # Random
            # 候補の中からランダムに選ぶ
            action = random.choice(options)
            # print(f"random action: {action}")

        return action
