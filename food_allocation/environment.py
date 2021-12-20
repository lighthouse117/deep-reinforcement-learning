# Environmentクラス

import numpy as np
from agent import Agent
from status import StockRemaining, StockChange
import matplotlib.pyplot as plt


FOODS = [20, 20, 20]
NUM_FOODS = len(FOODS)
MAX_UNITS = max(FOODS)

AGENTS_COUNT = 3

REQUESTS = [
    [10, 10, 10],
    [10, 10, 10],
    [10, 10, 10]
]

MAX_EPISODES = 100001
MAX_STEPS = 200

GREEDY_CYCLE = 100


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
        self.env_state = tuple([StockRemaining.MANY for i in range(
            NUM_FOODS)] + [StockChange.NONE for i in range(NUM_FOODS)])

        # self.print_env_state()
        states = []

        for agent in self.agents:
            agent_state = agent.reset(self.env_state)
            states.append(agent_state)
        return states

    def get_actions(self, states):
        actions = []
        # すべてのエージェントに対して
        for agent, state in zip(self.agents, states):
            # 行動を決定
            action = agent.decide_action(state)
            actions.append(action)
            if action == NUM_FOODS:
                # print(f"{agent.name} 行動: 何もしない")
                pass
            else:
                # print(f"{agent.name} 行動: 食品{action}を１つ取る")
                pass
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
                # print(f"{agent.name} 行動: 何もしない")
                pass
            else:
                # エージェントが食品を1つとる
                agent.get_food(action)
                # 本部の在庫が1つ減る
                self.stock[action] -= 1
        #         print(f"{agent.name} 行動: 食品{action}を１つ取る")
        #     print(f"{agent.name} 在庫:{agent.stock} 要求:{agent.REQUESTS}")
        # print(f"本部の在庫（更新前）: {old_stock}")
        # print(f"本部の在庫（更新後）: {self.stock}")

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
            # print("\n****** 終了条件を満たしています！ ******")
            reward = self.get_reward(greedy)
            states_next = [None] * AGENTS_COUNT
        else:
            # 終了時以外、報酬は0
            reward = 0
            states_next = []
            for agent in self.agents:
                states_next.append(agent.get_state(self.env_state))

        return actions, states_next, reward, done

    def get_env_state_next(self, old_stock):
        remaining = []
        change = []

        granularity = len(StockRemaining) - 2
        for amount, default in zip(self.stock, FOODS):
            section = round(default / granularity)
            if amount == 0:
                remaining.append(StockRemaining.NONE)
            elif amount < section:
                remaining.append(StockRemaining.FEW)
            elif amount < default:
                remaining.append(StockRemaining.MANY)
            else:
                remaining.append(StockRemaining.FULL)

        difference = old_stock - self.stock
        for diff in difference:
            if diff == 0:
                change.append(StockChange.NONE)
            elif diff == 1:
                change.append(StockChange.SLIGHT)
            elif diff == 2:
                change.append(StockChange.SOMEWHAT)
            else:
                change.append(StockChange.GREAT)
        # print(f"本部の在庫変動: {difference}")

        # print(f"本部在庫の変動: {diff}")

        state_next = tuple(remaining + change)
        # self.print_env_state()
        return state_next

    def learn(self, states, actions, reward, states_next):
        for agent, state, action, state_next in zip(self.agents, states, actions, states_next):
            agent.learn(state, action, reward, state_next)

    def get_reward(self, greedy):
        satisfactions = []
        for agent in self.agents:
            satisfaction = agent.get_satisfaction()
            satisfactions.append(satisfaction)
            if greedy:
                # print(f"{agent.name}の満足度: {satisfaction:.1f} %")
                pass

        deviation = np.std(np.array(satisfactions))

        remaining = np.sum(self.stock)

        # if deviation + remaining == 0:
        #     reward = 100
        # else:
        #     reward = (1 / (deviation + remaining * 100)) * 100

        reward = - (deviation + remaining)

        if greedy:
            # print(f"満足度の標準偏差: {deviation:.1f}")
            # print(f"食品の残り個数: {remaining}")
            print(f"報酬: {reward:.1f}")
            pass

        return reward

    def print_env_state(self):
        print("Env State: [", end="")
        for status in self.env_state:
            print(f"{status.name} ", end="")

        print("]")


def run():

    env = Environment()

    print(f"\n本部の在庫: {FOODS}")
    print(f"エージェント数: {AGENTS_COUNT}")
    print(f"エージェントの要求リスト: {REQUESTS}\n")

    result_reward = []

    for episode in range(MAX_EPISODES):

        if episode % GREEDY_CYCLE == 0:
            greedy = True
            print(
                f"-------------- Episode:{episode} (greedy) --------------")
        else:
            greedy = False
            # print(
            # f"-------------- Episode:{episode} --------------")

        states = env.reset()

        for step in range(MAX_STEPS):
            # print(f"\n------- Step:{step} -------")

            actions, states_next, reward, done = env.step(states, greedy)

            if step == MAX_STEPS - 1:
                print("最大ステップ数を超えました")
                reward = env.get_reward(greedy)

            if not greedy:
                env.learn(states, actions, reward, states_next)

            if done or step == MAX_STEPS - 1:
                break

            states = states_next

        if greedy:
            # print(f"要したステップ数: {step}")
            result_reward.append(reward)

    x = np.array(
        [i * GREEDY_CYCLE for i in range(len(result_reward))])
    y = np.array(result_reward)
    plt.plot(x, y)
    plt.xlim(0,)
    plt.ylim(top=0)
    # plt.title("")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    # plt.text(800, self.min_distance_history[0], "ε={}".format(EPSILON))
    plt.show()


if __name__ == "__main__":
    run()
