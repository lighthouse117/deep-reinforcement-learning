from state import State


class Agent:
    def __init__(self, num_states, num_actions):
        # 環境を生成
        self.state = State(num_states, num_actions)

    def update_Q_function(self, observation, action, reward, observation_next):
        # Qテーブルの更新
        self.state.update_Q_table(
            observation, action, reward, observation_next)

    def get_action(self, observation, step):
        # 行動
        action = self.state.decide_action(observation, step)
        return action
