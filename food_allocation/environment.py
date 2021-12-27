# Environmentクラス

from typing import List
import numpy as np
from agent import Agent
from status import StockRemaining, StockChange, Satisfaction
import matplotlib.pyplot as plt


FOODS = [20, 20, 20]
NUM_FOODS = len(FOODS)
MAX_UNITS = max(FOODS)

AGENTS_COUNT = 3

REQUESTS = [
    [10, 10, 10],
    [5, 10, 5],
    [5, 5, 10]
]

MAX_EPISODES = 101
MAX_STEPS = 200

GREEDY_CYCLE = 100

INITIAL_EPSILON = 1.0
MINIMUM_EPSILON = 0.5
EPSILON_DELTA = (INITIAL_EPSILON - MINIMUM_EPSILON) / (MAX_EPISODES * 0.9)

INITIAL_ALPHA = 0.5
MINIMUM_ALPHA = 0.01
ALPHA_DELTA = (INITIAL_ALPHA - MINIMUM_ALPHA) / (MAX_EPISODES * 0.9)


class Environment:

    def __init__(self):
        state_size = pow(len(StockRemaining), NUM_FOODS) * pow(len(StockChange),
                                                               NUM_FOODS) * pow(len(Satisfaction), NUM_FOODS)
        print(f"状態数: {state_size}")
        self.agents = self.init_agents()

    def init_agents(self):
        agents: List[Agent] = []
        for i in range(AGENTS_COUNT):
            name = f"Agent{i + 1}"
            agent = Agent(
                name, MAX_UNITS, NUM_FOODS, np.array(REQUESTS[i]))
            agents.append(agent)
        return agents

    def reset(self):
        self.stock = np.array(FOODS, dtype=np.int64)
        states = [None] * AGENTS_COUNT

        for agent in self.agents:
            agent.reset(self.stock)

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

    def step(self, old_states, greedy, epsilon, episode):
        states = []
        actions = []
        rewards = []

        # すべてのエージェントに対して
        for agent in self.agents:

            state = agent.observe_state(self.stock, FOODS)
            states.append(state)

            # 行動を決定
            action = agent.decide_action(
                state, self.stock, greedy, epsilon)

            if greedy and episode == MAX_EPISODES - 1:
                print(f"本部の在庫: {self.stock}")
                agent.print_state(state)

                diff = agent.stock - agent.REQUESTS
                if action == NUM_FOODS:
                    action_string = "何もしない"
                else:
                    action_string = f"食品{action}"

                print(
                    f"{agent.brain.Q[state]} 要求との差:{diff} 行動: {action_string}")

            actions.append(action)

            if action != NUM_FOODS:
                # エージェントが食品を1つとる
                agent.get_food(action)
                # 本部の在庫が1つ減る
                self.stock[action] -= 1

                # if agent.current_requests[action] == 0:
                #     rewards.append(-50)
                # else:
                #     rewards.append(0)

            # 分配終了確認
            done = self.check_done()

            # if done:

            # if greedy and episode == MAX_EPISODES - 1:
            #     print(f"本部の在庫（更新前）: {old_stock}")
            #     print(f"本部の在庫（更新後）: {self.stock}")

            # 次の状態へ遷移
            # self.env_state = self.get_env_state_next(old_stock)

        if done:
            # 終了時は各エージェントの報酬を計算
            # print("\n****** 終了条件を満たしています！ ******")
            rewards = [self.get_reward(greedy)] * AGENTS_COUNT
        else:
            # 終了時以外、報酬は0
            rewards = [0] * AGENTS_COUNT

        return actions, states, rewards, done

    def check_done(self):
        # 終了条件を満たしているかチェック
        all_agent_done = True
        for agent in self.agents:
            if not agent.done:
                all_agent_done = False
                break

        # 在庫がすべてなくなったか、全エージェントの取れる行動がなくなったか
        done = np.all(self.stock == 0) or all_agent_done

        return done

    def learn(self, states, actions, rewards, states_next, alpha):
        for agent, state, action, reward, state_next in zip(self.agents, states, actions, rewards, states_next):
            agent.learn(state, action, reward, state_next, alpha)

    def get_reward(self, greedy):
        loss_values = []
        for agent in self.agents:
            loss = agent.get_loss_value()
            loss_values.append(loss)
            # diff = agent.stock - agent.REQUESTS
            # if greedy:
            #     print(f"{agent.name}: 損失{loss:.1f} 要求との差{diff} 在庫{agent.stock}")
            #     pass

        average = np.average(np.array(loss_values))
        deviation = np.std(np.array(loss_values))

        remaining = np.sum(self.stock)

        # if deviation + remaining == 0:
        #     reward = 100
        # else:
        #     reward = (1 / (deviation + remaining * 100)) * 100

        reward = - (average + deviation) * 0.1 - remaining * 10

        if greedy:
            print(f"損失の平均: {average:.1f}")
            print(f"損失の標準偏差: {deviation:.1f}")
            print(f"食品の残り個数: {remaining}")
            print(f"報酬: {reward:.2f}")
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

    result_reward_x = []
    result_reward_y = []

    result_optimal_reward_x = []
    result_optimal_reward_y = []

    # result_epsilon = []

    optimal_reward = -10000
    optimal_agent_stock = []
    optimal_loss_values = []

    epsilon = INITIAL_EPSILON
    alpha = INITIAL_ALPHA

    fig, ax = plt.subplots()

    # plt.xlim(0,)
    # plt.ylim(top=0)

    # plt.title("")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    # plt.text(800, self.min_distance_history[0], "ε={}".format(EPSILON))

    lines1, = ax.plot([], [], zorder=2)
    # lines2, = plt.plot(x2, y2, zorder=1)

    for episode in range(MAX_EPISODES):

        if episode % GREEDY_CYCLE == 0:
            greedy = True
            print(
                f"-------------- Episode:{episode} (greedy) --------------")
            # print(f"EPSILON: {epsilon:.3f}")
            # print(f"ALPHA: {alpha:.3f}")
        else:
            greedy = False
            # print(
            #     f"-------------- Episode:{episode} --------------")
        states = []
        actions = []
        rewards = []

        old_states = env.reset()
        old_actions = []
        old_rewards = []

        step = 0

        while True:
            if greedy and episode == MAX_EPISODES - 1:
                print(f"\n------- Step:{step} -------")

            for agent in env.agents:
                state = agent.observe_state(env.stock, FOODS)
                states.append(state)

                # 行動を決定
                action = agent.decide_action(
                    state, env.stock, greedy, epsilon)

                if greedy and episode == MAX_EPISODES - 1:
                    print(f"本部の在庫: {env.stock}")
                    agent.print_state(state)

                    diff = agent.stock - agent.REQUESTS
                    if action == NUM_FOODS:
                        action_string = "何もしない"
                    else:
                        action_string = f"食品{action}"

                    print(
                        f"{agent.brain.Q[state]} 要求との差:{diff} 行動: {action_string}")

                actions.append(action)

                if action != NUM_FOODS:
                    # エージェントが食品を1つとる
                    agent.get_food(action)
                    # 本部の在庫が1つ減る
                    env.stock[action] -= 1

                # 分配終了確認
                done = env.check_done()

            if done:

                # if greedy and episode == MAX_EPISODES - 1:
                #     print(f"本部の在庫（更新前）: {old_stock}")
                #     print(f"本部の在庫（更新後）: {self.stock}")

                # 次の状態へ遷移
                # self.env_state = self.get_env_state_next(old_stock)
                break

            if done:
                # 終了時は各エージェントの報酬を計算
                # print("\n****** 終了条件を満たしています！ ******")
                rewards = [env.get_reward(greedy)] * AGENTS_COUNT
            else:
                # 終了時以外、報酬は0
                rewards = [0] * AGENTS_COUNT

            if step == MAX_STEPS - 1:
                rewards = [env.get_reward(greedy)] * AGENTS_COUNT
                if greedy:
                    print("最大ステップ数を超えました")

            if old_states is not None:
                env.learn(old_states, old_actions, old_rewards, states, alpha)

            if done or step == MAX_STEPS - 1:
                break

            old_states = states
            old_actions = actions
            old_rewards = rewards

        if rewards[0] > optimal_reward:
            optimal_reward = rewards[0]
            optimal_episode = episode
            print(f"\n最大の報酬を更新: {optimal_reward}")
            optimal_loss_values = []
            optimal_agent_stock = []
            for agent in env.agents:
                loss = agent.get_loss_value()
                stock = agent.stock
                optimal_loss_values.append(loss)
                optimal_agent_stock.append(stock)
                print(f"{agent.name}: 損失{loss:.1f} 在庫{stock}")
            print()

        if epsilon > MINIMUM_EPSILON:
            epsilon = epsilon - EPSILON_DELTA
        if alpha > MINIMUM_ALPHA:
            alpha = alpha - ALPHA_DELTA

        if greedy:
            print(f"要したステップ数: {step}")
            # print(f"現在のエピソード: {episode}")
            result_reward_x.append(episode)
            result_reward_y.append(rewards[0])
            result_optimal_reward_x.append(episode)
            result_optimal_reward_y.append(optimal_reward)
            lines1.set_data(result_reward_x, result_reward_y)

            ax.relim()
            ax.autoscale_view()

            # lines2.set_data(result_optimal_reward_x, result_optimal_reward_y)
            # plt.pause(0.001)

    print("\n---- 最適解 ----")
    for agent, loss, stock, request in zip(env.agents, optimal_loss_values, optimal_agent_stock, REQUESTS):
        diff = stock - request
        print(f"{agent.name}: 損失{loss:.1f} 要求との差{diff} 在庫{stock}")
    print(f"報酬: {optimal_reward:.4f}")
    print(f"発見したエピソード: {optimal_episode}")

    # plt.show()


if __name__ == "__main__":
    run()
