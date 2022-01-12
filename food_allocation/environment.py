# Environmentクラス

from typing import List
import numpy as np
from agent import Agent
from status import StockRemaining, StockChange, Satisfaction
import matplotlib.pyplot as plt
import copy


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
MAX_STEPS = 100

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
        states = None

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

    # def step(self, old_states, greedy, epsilon, episode):
    #     states = []
    #     actions = []
    #     rewards = []

    #     # すべてのエージェントに対して
    #     for agent in self.agents:

    #         state = agent.observe_state(self.stock, FOODS)
    #         states.append(state)

    #         # 行動を決定
    #         action = agent.decide_action(
    #             state, self.stock, greedy, epsilon)

    #         if greedy and episode == MAX_EPISODES - 1:
    #             print(f"本部の在庫: {self.stock}")
    #             agent.print_state(state)

    #             diff = agent.stock - agent.REQUESTS
    #             if action == NUM_FOODS:
    #                 action_string = "何もしない"
    #             else:
    #                 action_string = f"食品{action}"

    #             print(
    #                 f"{agent.brain.Q[state]} 要求との差:{diff} 行動: {action_string}")

    #         actions.append(action)

    #         if action != NUM_FOODS:
    #             # エージェントが食品を1つとる
    #             agent.get_food(action)
    #             # 本部の在庫が1つ減る
    #             self.stock[action] -= 1

    #             # if agent.current_requests[action] == 0:
    #             #     rewards.append(-50)
    #             # else:
    #             #     rewards.append(0)

    #         # 分配終了確認
    #         done = self.check_done()

    #         # if done:

    #         # if greedy and episode == MAX_EPISODES - 1:
    #         #     print(f"本部の在庫（更新前）: {old_stock}")
    #         #     print(f"本部の在庫（更新後）: {self.stock}")

    #         # 次の状態へ遷移
    #         # self.env_state = self.get_env_state_next(old_stock)

    #     if done:
    #         # 終了時は各エージェントの報酬を計算
    #         # print("\n****** 終了条件を満たしています！ ******")
    #         rewards = self.get_rewards()
    #     else:
    #         # 終了時以外、報酬は0
    #         rewards = [0] * AGENTS_COUNT

    #     return actions, states, rewards, done

    def check_food_run_out(self):
        # 全ての在庫が0になったかチェック
        return np.all(self.stock == 0)

    def check_all_agent_done(self):
        # 全てのエージェントが終了条件を満たしているかチェック
        all_agent_done = True
        for agent in self.agents:
            if not agent.done:
                all_agent_done = False
                break

        # 全エージェントの取れる行動がなくなったか
        return all_agent_done

    def learn(self, states, actions, rewards, states_next, alpha):
        for agent, state, action, reward, state_next in zip(self.agents, states, actions, rewards, states_next):
            agent.learn(state, action, reward, state_next, alpha)

    def get_rewards(self, greedy):
        rewards = []
        satisfactions = []

        remaining = np.sum(self.stock)

        for agent in self.agents:
            satisfaction = agent.get_satisfaction()
            satisfactions.append(satisfaction)

        average = np.average(satisfactions)

        if greedy:
            # print(f"損失の標準偏差: {deviation:.1f}")
            print(f"満足度の平均: {average:.2f}")
            print(f"食品の残り個数: {remaining}")
            pass
        for agent, satisfaction in zip(self.agents, satisfactions):
            abs_deviation = np.absolute(average - satisfaction)
            reward = - (abs_deviation + remaining * 10)
            rewards.append(reward)
            if greedy:
                print(
                    f"{agent.name}: 報酬{reward:.3f} 絶対偏差{abs_deviation:.1f} 満足度{satisfaction:.1f} 要求{agent.REQUESTS} 在庫{agent.stock}")
                pass

        # average = np.average(np.array(satisfactions))
        # deviation = np.std(np.array(loss_values))

        # if deviation + remaining == 0:
        #     reward = 100
        # else:
        #     reward = (1 / (deviation + remaining * 100)) * 100

        # reward = - deviation * 0.1 - remaining * 10

        return rewards

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

    result_agent_x = []
    results_agents_y = [[], [], []]

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

    line1, = ax.plot([], [])
    line2, = ax.plot([], [])
    line3, = ax.plot([], [])

    lines = [line1, line2, line3]

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
        states = [[], [], []]
        actions = [[], [], []]
        rewards = []

        old_states = env.reset()
        old_actions = []
        old_rewards = []

        all_agent_done = False
        step = 0

        while True:
            if greedy and episode == MAX_EPISODES - 1:
                print(f"\n------- Step:{step} -------")

            for index, agent in enumerate(env.agents):
                if greedy and episode == MAX_EPISODES - 1:
                    print(f"本部の在庫: {env.stock}")
                    diff = agent.old_env_stock - env.stock
                    print(f"{agent.name} 前回からの変化:{diff}")

                # 状態を観測
                state = agent.observe_state(env.stock, FOODS)
                states[index] = state

                # 行動を決定
                action = agent.decide_action(
                    state, env.stock, greedy, epsilon)
                actions[index] = action

                if greedy and episode == MAX_EPISODES - 1:
                    agent.print_state(state)
                    # print(state)
                    if action == NUM_FOODS:
                        action_string = "何もしない"
                    else:
                        action_string = f"食品{action}"

                    print(
                        f"Q:{agent.brain.Q[state]} 行動: {action_string}")

                # 行動をとる
                if action != NUM_FOODS:
                    # エージェントが食品を1つとる
                    agent.get_food(action)
                    # 本部の在庫が1つ減る
                    env.stock[action] -= 1

                # 分配終了確認
                food_done = env.check_food_run_out()
                if agent.done or food_done:
                    if greedy:
                        print(f"\n{agent.name} ****** 終了条件を満たしています! ******")
                    # TODO: 学習
                    reward =

                all_agent_done = env.check_all_agent_done()
                if all_agent_done:
                    break

            if all_agent_done:
                # 終了時は各エージェントの報酬を計算
                # print("\n****** 終了条件を満たしています！ ******")
                rewards = env.get_rewards(greedy)
                if greedy:
                    print(rewards)

            else:
                # 終了時以外、報酬は0
                rewards = [0] * AGENTS_COUNT

            if step == MAX_STEPS - 1:
                rewards = env.get_rewards(greedy)
                if greedy:
                    print("最大ステップ数を超えました")

            if old_states is not None:
                env.learn(old_states, old_actions, rewards, states, alpha)

            if done or step == MAX_STEPS - 1:
                break

            old_states = copy.deepcopy(states)
            old_actions = copy.deepcopy(actions)
            # old_rewards = copy.deepcopy(rewards)

            step += 1

            # print(states)

        # best_reward = np.amin(n)
        # if rewards[0] > optimal_reward:
        #     optimal_reward = rewards[0]
        #     optimal_episode = episode
        #     print(f"\n最大の報酬を更新: {optimal_reward}")
        #     optimal_loss_values = []
        #     optimal_agent_stock = []
        #     for agent in env.agents:
        #         loss = agent.get_loss_value()
        #         stock = agent.stock
        #         optimal_loss_values.append(loss)
        #         optimal_agent_stock.append(stock)
        #         print(f"{agent.name}: 損失{loss:.1f} 在庫{stock}")
        #     print()

        if epsilon > MINIMUM_EPSILON:
            epsilon = epsilon - EPSILON_DELTA
        if alpha > MINIMUM_ALPHA:
            alpha = alpha - ALPHA_DELTA

        if greedy:
            print(f"要したステップ数: {step}")
            # print(f"現在のエピソード: {episode}")
            # result_reward_x.append(episode)
            # result_reward_y.append(rewards[0])
            # result_optimal_reward_x.append(episode)
            # result_optimal_reward_y.append(optimal_reward)
            # lines1.set_data(result_reward_x, result_reward_y)
            result_agent_x.append(episode)

            for agent, line, result in zip(env.agents, lines, results_agents_y):
                result.append(agent.get_satisfaction())
                line.set_data(result_agent_x, result)

            # lines1.set_data(result_agents_x, result_agent1_y)
            # lines2.set_data(result_agents_x, result_agent2_y)
            # lines3.set_data(result_agents_x, result_agent3_y)

            ax.relim()
            ax.autoscale_view()

            # lines1.set_data(result_satisfaction_x, result_satisfaction_y)

            # lines2.set_data(result_optimal_reward_x, result_optimal_reward_y)
            plt.pause(0.001)

    # print("\n---- 最適解 ----")
    # for agent, loss, stock, request in zip(env.agents, optimal_loss_values, optimal_agent_stock, REQUESTS):
    #     diff = stock - request
    #     print(f"{agent.name}: 損失{loss:.1f} 要求との差{diff} 在庫{stock}")
    # print(f"報酬: {optimal_reward:.4f}")
    # print(f"発見したエピソード: {optimal_episode}")

    plt.show()


if __name__ == "__main__":
    run()
