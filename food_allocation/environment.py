# Environmentクラス

import numpy as np
from agent import Agent


FOODS = [6, 3, 5]
NUM_FOODS = len(FOODS)
MAX_UNITS = max(FOODS)

AGENTS_COUNT = 2

REQUESTS = [
    [5, 2, 3],
    [5, 1, 3],
]

MAX_EPISODES = 50
MAX_STEPS = 40

GREEDY_CYCLE = 10


class Environment:

    def __init__(self):
        self.agents = self.init_agents()

    def init_agents(self):
        agents = []
        for i in range(AGENTS_COUNT):
            name = f"Agent{i + 1}"
            agent = Agent(name, MAX_UNITS, NUM_FOODS, np.array(REQUESTS[i]))
            agents.append(agent)
        return agents

    def reset(self):
        self.stock = np.array(FOODS, dtype=np.int64)
        self.env_state = tuple(np.concatenate(
            [self.stock, np.zeros(NUM_FOODS, dtype=np.int64)]))
        print(f"env_state: {self.env_state}")
        states = []
        for agent in self.agents:
            agent.reset()
            states.append(
                self.env_state + tuple(np.zeros(NUM_FOODS, dtype=np.int64)))
        print(f"states: {states}")
        return states

    def get_actions(self, states):
        actions = []
        # すべてのエージェントに対して
        for agent, state in zip(self.agents, states):
            # 行動を決定
            action = agent.decide_action(state)
            actions.append(action)
            if action == NUM_FOODS:
                print(f"{agent.name} 行動: 何もしない")
            else:
                print(f"{agent.name} 行動: 食品{action}を１つ取る")
        return actions

    def step(self, states, greedy):
        old_stock = self.stock.copy()
        actions = []

        # すべてのエージェントに対して
        for agent, state in zip(self.agents, states):
            # 行動を決定
            action = agent.decide_action(state, self.stock, greedy)
            actions.append(action)
            if action == NUM_FOODS:
                print(f"{agent.name} 行動: 何もしない")
            else:
                # エージェントが食品を1つとる
                agent.get_food(action)
                # 本部の在庫が1つ減る
                self.stock[action] -= 1
                print(f"{agent.name} 行動: 食品{action}を１つ取る")
            print(f"{agent.name} 手元の在庫:{agent.stock} 要求リスト:{agent.current_requests}")
        print(f"本部の在庫（更新前）: {old_stock}")
        print(f"本部の在庫（更新後）: {self.stock}")

        # 次の状態へ遷移
        self.env_state = self.get_env_state_next(old_stock)

        # 終了条件を満たしているかチェック
        all_agent_done = True
        for agent in self.agents:
            if not agent.done:
                all_agent_done = False
                break

        # 在庫がすべてなくなったか、全エージェントの取れる行動がなくなったか
        done = np.all(self.stock == 0) or all_agent_done

        if done:
            # 終了時に各エージェントの報酬を計算
            print("\n****** 終了条件を満たしています！ ******")
            reward = self.get_reward()
            states_next = [None] * AGENTS_COUNT
        else:
            # 終了時以外、報酬は0
            reward = 0
            states_next = []
            for agent in self.agents:
                states_next.append(agent.get_state(self.env_state))

        return actions, states_next, reward, done

    def get_env_state_next(self, old_stock):
        diff = self.stock - old_stock
        # print(f"本部在庫の変動: {diff}")

        diff += MAX_UNITS

        state_next = tuple(np.concatenate(
            [self.stock, diff]))
        return state_next

    def learn(self, states, actions, reward, states_next):
        for agent, state, action, state_next in zip(self.agents, states, actions, states_next):
            agent.learn(state, action, reward, state_next)

    def get_reward(self):
        satisfactions = []
        for agent in self.agents:
            satisfactions.append(agent.get_satisfaction())

        deviation = np.std(np.array(satisfactions))
        print(f"\n満足度の標準偏差: {deviation:.1f}")

        remaining = np.sum(self.stock)
        print(f"食品の残り個数: {remaining}")

        reward = -(deviation + remaining)
        print(f"\n報酬: {reward:.1f}\n")

        return reward


def run():

    env = Environment()

    print(f"\n本部の在庫: {FOODS}")
    print(f"エージェント数: {AGENTS_COUNT}")
    print(f"エージェントの要求リスト: {REQUESTS}\n")

    for episode in range(MAX_EPISODES):
        greedy = False
        if episode % GREEDY_CYCLE == 0 and episode != 0:
            greedy = True
            print(
                f"\n-------------- Episode:{episode} (greedy) --------------")
        else:
            print(
                f"-------------- Episode:{episode} --------------")

        states = env.reset()

        for step in range(MAX_STEPS):
            print(f"\n------- Step:{step} -------")

            actions, states_next, reward, done = env.step(states, greedy)

            env.learn(states, actions, reward, states_next)

            states = states_next

            if done:
                break

            else:
                if step == MAX_STEPS - 1:
                    print("\n最大ステップ数を超えました")


if __name__ == "__main__":
    run()
