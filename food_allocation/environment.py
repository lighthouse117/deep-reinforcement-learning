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
    [5, 5, 10],
]

MAX_EPISODES = 100001
MAX_STEPS = 100

GREEDY_CYCLE = 1000

INITIAL_EPSILON = 1.0
MINIMUM_EPSILON = 0.5
EPSILON_DELTA = (INITIAL_EPSILON - MINIMUM_EPSILON) / (MAX_EPISODES * 0.95)

INITIAL_ALPHA = 0.5
MINIMUM_ALPHA = 0.01
ALPHA_DELTA = (INITIAL_ALPHA - MINIMUM_ALPHA) / (MAX_EPISODES * 0.95)


class Environment:

    def __init__(self, f):
        self.f = f
        state_size = pow(len(StockRemaining), NUM_FOODS) * pow(len(StockChange),
                                                               NUM_FOODS) * pow(len(Satisfaction), NUM_FOODS)

        print(f"状態数: {state_size}", file=self.f)
        self.agents = self.init_agents()

    def init_agents(self):
        agents: List[Agent] = []
        for i in range(AGENTS_COUNT):
            name = f"Agent{i + 1}"
            agent = Agent(
                name, MAX_UNITS, NUM_FOODS, np.array(REQUESTS[i]), self.f)
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

    def check_agents_food_done(self):
        # 全てのエージェントが終了条件を満たしているかチェック
        agents_done = True
        for agent in self.agents:
            if not agent.food_done:
                agents_done = False
                break
        # 全エージェントの取れる行動がなくなったか
        return agents_done

    def check_agents_learning_done(self):
        agents_done = True
        for agent in self.agents:
            if not agent.learning_done:
                agents_done = False
                break
        return agents_done

    def learn(self, states, actions, rewards, states_next, alpha):
        for agent, state, action, reward, state_next in zip(self.agents, states, actions, rewards, states_next):
            agent.learn(state, action, reward, state_next, alpha)

    def get_reward(self, target_agent: Agent, terminal, greedy):
        if terminal:
            satisfactions = []

            remaining = np.sum(self.stock)

            for agent in self.agents:
                s = agent.get_satisfaction()
                satisfactions.append(s)
            average = np.average(satisfactions)

            if greedy:
                # print(f"損失の標準偏差: {deviation:.1f}")
                print(f"満足度の平均: {average:.2f}", file=self.f)
                print(f"食品の残り個数: {remaining}", file=self.f)
                pass

            agent_satisfaction = target_agent.get_satisfaction()
            abs_deviation = np.absolute(average - agent_satisfaction)

            reward = - (abs_deviation + remaining * 10)

            if greedy:
                print(
                    f"{target_agent.name}: 報酬{reward:.3f} 絶対偏差{abs_deviation:.1f} 満足度{agent_satisfaction:.1f} 要求{target_agent.REQUESTS} 在庫{target_agent.stock}", file=self.f)
                pass
        else:
            reward = 0

        return reward

    def print_env_state(self):
        print("Env State: [", end="", file=self.f)
        for status in self.env_state:
            print(f"{status.name} ", end="", file=self.f)

        print("]", file=self.f)


def run():

    path = "food_allocation/results/log.txt"
    f = open(path, mode="w", encoding="UTF-8")

    env = Environment(f)

    print(f"\n本部の在庫: {FOODS}", file=f)
    print(f"エージェント数: {AGENTS_COUNT}", file=f)
    print(f"エージェントの要求リスト: {REQUESTS}\n", file=f)

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

    line1, = ax.plot([], [], label="Agent1")
    line2, = ax.plot([], [], label="Agent2")
    line3, = ax.plot([], [], label="Agent3")

    ax.legend()

    lines = [line1, line2, line3]

    # lines2, = plt.plot(x2, y2, zorder=1)

    # 各エピソード
    for episode in range(MAX_EPISODES):

        if episode % GREEDY_CYCLE == 0:
            greedy = True
            print(
                f"-------------- Episode:{episode} (greedy) --------------")
            print(
                f"-------------- Episode:{episode} (greedy) --------------", file=f)
            # print(f"EPSILON: {epsilon:.3f}")
            # print(f"ALPHA: {alpha:.3f}")
        else:
            greedy = False
            # print(
            #     f"-------------- Episode:{episode} --------------")
        states = [[], [], []]
        actions = [[], [], []]
        rewards = [0, 0, 0]

        old_states = env.reset()
        old_actions = []

        learning_complete = False
        step = 0

        # 各ステップ
        while True:
            if greedy:
                print(f"\n------- Step:{step} -------", file=f)

            # 各エージェント
            for index, agent in enumerate(env.agents):
                if greedy:
                    print(f"本部の在庫: {env.stock}", file=f)
                    diff = agent.old_env_stock - env.stock
                    print(f"{agent.name} 前回からの変化:{diff}", file=f)

                # 状態を観測
                state = agent.observe_state(env.stock, FOODS)
                states[index] = state

                # 行動を決定
                action = agent.decide_action(
                    state, env.stock, greedy, epsilon)
                actions[index] = action

                if greedy:
                    agent.print_state(state)
                    # print(state)
                    if action == NUM_FOODS:
                        action_string = "何もしない"
                    else:
                        action_string = f"食品{action}"

                    print(
                        f"Q:{agent.brain.Q[state]} 行動: {action_string}", file=f)

                # 行動をとる
                if action != NUM_FOODS:
                    # エージェントが食品を1つとる
                    agent.get_food(action)
                    # 本部の在庫が1つ減る
                    env.stock[action] -= 1

                # 終了条件確認
                food_done = env.check_food_run_out()
                all_agents_done = env.check_agents_food_done()
                exceed_max_step = step == MAX_STEPS - 1

                episode_terminal = food_done or all_agents_done or exceed_max_step

                reward = env.get_reward(agent, episode_terminal, greedy)

                if old_states is not None:
                    agent.learn(
                        old_states[index], old_actions[index], reward, state, alpha)

                if episode_terminal:
                    if greedy:
                        print("*** 終了条件を満たしています ***", file=f)
                        print(f"{agent.name} 学習終了\n", file=f)
                    rewards[index] = reward
                    agent.learning_done = True
                    learning_complete = env.check_agents_learning_done()
                    if learning_complete:
                        break

            if exceed_max_step and greedy:
                print("最大ステップ数を超えました", file=f)

            if learning_complete:
                break

            old_states = copy.deepcopy(states)
            old_actions = copy.deepcopy(actions)

            step += 1

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
            print(f"要したステップ数: {step}", file=f)
            # print(f"現在のエピソード: {episode}")
            # result_reward_x.append(episode)
            # result_reward_y.append(rewards[0])
            # result_optimal_reward_x.append(episode)
            # result_optimal_reward_y.append(optimal_reward)
            # lines1.set_data(result_reward_x, result_reward_y)
            result_agent_x.append(episode)

            for i, agent, line, result in zip(range(AGENTS_COUNT), env.agents, lines, results_agents_y):
                result.append(rewards[i])
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
