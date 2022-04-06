import numpy as np

# 各状態の分割数
NUM_DIZITIZED = 6

# 学習パラメータ
GAMMA = 0.99  # 時間割引率
ETA = 0.5  # 学習係数


class State:
    def __init__(self, num_states, num_actions):
        # 行動数を取得
        self.num_actions = num_actions

        # Qテーブルを作成　(分割数^状態数)×(行動数)
        self.q_table = np.random.uniform(low=-1, high=1, size=(
            NUM_DIZITIZED**num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        # 観測した状態デジタル変換する閾値を求める
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def analog2digitize(self, observation):
        # 状態の離散化
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIZITIZED)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIZITIZED)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIZED)),
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIZITIZED))
        ]
        return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])

    def update_Q_table(self, observation, action, reward, observation_next):
        # 状態の離散化
        state = self.analog2digitize(observation)
        state_next = self.analog2digitize(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        # Qテーブルを更新(Q学習)
        self.q_table[state, action] = self.q_table[state, action] + \
            ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        # ε-greedy法で行動を選択する
        state = self.analog2digitize(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            # 最も価値の高い行動を行う。
            action = np.argmax(self.q_table[state][:])
        else:
            # 適当に行動する。
            action = np.random.choice(self.num_actions)
        return action
