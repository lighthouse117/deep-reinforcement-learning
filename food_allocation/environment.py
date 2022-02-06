# Environmentクラス
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime
import os


from agent import Agent
from status import StockRemaining, StockChange, Satisfaction, Progress
from config import LearningParameters as lp
from config import EnvironmentSettings as es


class Environment:

    def __init__(self, f):
        self.f = f
        # state_size = pow(len(StockRemaining), es.NUM_FOODS) * \
        #     pow(len(Satisfaction), es.NUM_FOODS) * len(Progress)

        state_size = pow(len(StockRemaining), es.NUM_FOODS) * pow(len(StockChange),
                                                                  es.NUM_FOODS) * pow(len(Satisfaction), es.NUM_FOODS) * len(Progress)
        action_size = es.NUM_FOODS + 1

        print("========= 各エージェントのもつ状態行動空間 =========", file=self.f)
        print(f"状態数: {state_size:,}", file=self.f)
        print(f"行動数: {action_size}", file=self.f)
        print(f"状態 × 行動の組み合わせ: {(state_size * action_size):,}", file=self.f)
        print("\n\n", file=self.f)
        self.agents = self.init_agents()

    def init_agents(self):
        agents: List[Agent] = []
        for i in range(es.AGENTS_COUNT):
            name = f"Agent{i + 1}"
            agent = Agent(
                name, np.array(es.REQUESTS[i]), self.f)
            agents.append(agent)
        return agents

    def reset(self, greedy):
        self.stock = np.array(es.FOODS, dtype=np.int64)
        states = None

        for agent in self.agents:
            agent.reset(self.stock, greedy)

        return states

    def get_actions(self, states):
        actions = []
        # すべてのエージェントに対して
        for agent, state in zip(self.agents, states):
            # 行動を決定
            action = agent.decide_action(state)
            actions.append(action)
            if action == es.NUM_FOODS:
                # print(f"{agent.name} 行動: 何もしない")
                pass
            else:
                # print(f"{agent.name} 行動: 食品{action}を１つ取る")
                pass
        return actions

    def check_food_run_out(self):
        # 全ての在庫が0になったかチェック
        return np.all(self.stock == 0)

    def check_agents_food_done(self):
        # 全てのエージェントが終了条件を満たしているかチェック
        all_done = True
        for agent in self.agents:
            if not agent.food_done:
                all_done = False
                break
        # 全エージェントの取れる行動がなくなったか
        return all_done

    def check_agents_learning_done(self):
        all_done = True
        for agent in self.agents:
            if not agent.learning_done:
                all_done = False
                break
        return all_done

    def learn(self, states, actions, rewards, states_next, alpha):
        for agent, state, action, reward, state_next in zip(self.agents, states, actions, rewards, states_next):
            agent.learn(state, action, reward, state_next, alpha)

    def get_reward(self, target_agent: Agent, terminal, greedy):
        if terminal:
            # satisfactions = []

            # remaining = np.sum(self.stock)

            # for agent in self.agents:
            #     s = agent.get_satisfaction()
            #     satisfactions.append(s)
            # average = np.average(satisfactions)

            # if greedy:
            #     # print(f"損失の標準偏差: {deviation:.1f}")
            #     print(f"満足度の平均: {average:.2f}", file=self.f)
            #     print(f"食品の残り個数: {remaining}", file=self.f)
            #     pass

            # target_satisfaction = target_agent.satisfaction
            # abs_deviation = np.absolute(average - target_satisfaction)

            # reward = - (abs_deviation + remaining * 10)
            # reward = - (abs_deviation + remaining)

            # print(abs_deviation + remaining)
            # if (abs_deviation + remaining) == 0.0:
            #     reward = 10
            # else:

            #     reward = 1 / (abs_deviation + remaining)

            reward = - target_agent.get_violation()

            if greedy:
                # print(
                #     f"{target_agent.name}: 報酬{reward:.3f} 絶対偏差{abs_deviation:.1f} 満足度{target_satisfaction:.1f} 要求{target_agent.REQUEST} 在庫{target_agent.stock}", file=self.f)

                print(
                    f"{target_agent.name}: 報酬{reward:.3f}  要求{target_agent.REQUEST} 在庫{target_agent.stock}", file=self.f)

        else:
            reward = -1

        return reward

    def print_env_state(self):
        print("Env State: [", end="", file=self.f)
        for status in self.env_state:
            print(f"{status.name} ", end="", file=self.f)

        print("]", file=self.f)


def run():
    np.set_printoptions(precision=5, floatmode='maxprec_equal')
    np.set_printoptions(suppress=True)

    DIR_PATH = "D:\\Lighthouse\\Documents\\Reinforcement Learning\\Food Distribution\\results"

    start_time = datetime.datetime.now()
    file_name_time = "{0:%Y-%m-%d_%H%M%S}".format(start_time)

    log_file_name = f"log_{file_name_time}.txt"
    path = os.path.join(DIR_PATH, "logs", log_file_name)
    f = open(path, mode="w", encoding="UTF-8")

    print("開始時刻: {0:%Y/%m/%d %H:%M:%S}\n".format(start_time), file=f)

    print("====================== 環境設定 ======================", file=f)
    print(f"食品の種類: {es.NUM_FOODS}", file=f)
    print(f"本部の在庫: {es.FOODS}", file=f)
    print(f"エージェント数: {es.AGENTS_COUNT}", file=f)
    print(f"エージェントの要求リスト: {es.REQUESTS}", file=f)
    print("\n\n", file=f)

    print("==================== 学習パラメーター ====================", file=f)
    print(f"エピソード数: {lp.MAX_EPISODES:,}", file=f)
    print(f"最大ステップ数: {lp.MAX_STEPS:,}\n", file=f)
    print(f"割引率: {lp.GAMMA}\n", file=f)
    print(f"学習率の初期値: {lp.INITIAL_ALPHA}", file=f)
    print(f"学習率の最終値: {lp.MINIMUM_ALPHA}", file=f)
    print(f"1エピソードごとの学習率の減少値: {lp.ALPHA_DELTA:.7f}\n", file=f)
    print(f"εの初期値: {lp.INITIAL_EPSILON}", file=f)
    print(f"εの最終値: {lp.MINIMUM_EPSILON}", file=f)
    print(f"1エピソードごとのεの減少値: {lp.EPSILON_DELTA:.7f}", file=f)
    print("\n\n", file=f)

    env = Environment(f)

    # result_reward_x = []
    # result_reward_y = []

    # result_optimal_reward_x = []
    # result_optimal_reward_y = []

    result_agent_x = []
    results_agents_y = [[], [], []]

    result_ave = []
    result_dev = []

    # result_epsilon = []

    # optimal_reward = -10000
    # optimal_agent_stock = []
    # optimal_loss_values = []

    epsilon = lp.INITIAL_EPSILON
    alpha = lp.INITIAL_ALPHA

    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.set_xlabel("Episode")
    ax2.set_xlabel("Episode")
    ax3.set_xlabel("Episode")

    ax1.set_ylabel("Reward")
    ax2.set_ylabel("Average")
    ax3.set_ylabel("Deviation")

    # plt.xlim(0,)
    # plt.ylim(top=0)

    # plt.title("")
    # plt.text(800, self.min_distance_history[0], "ε={}".format(EPSILON))

    line1, = ax1.plot([], [], label="Agent1")
    line2, = ax1.plot([], [], label="Agent2")
    line3, = ax1.plot([], [], label="Agent3")

    line_ave, = ax2.plot([], [])
    line_dev, = ax3.plot([], [])

    ax1.legend()

    lines = [line1, line2, line3]

    # 各エピソード
    for episode in range(lp.MAX_EPISODES):

        if episode % lp.GREEDY_CYCLE == 0:
            greedy = True
            print(
                f"-------------- Episode:{episode} (greedy) --------------")
            print(
                f"\n\n-------------- Episode:{episode} (greedy) ----------------------------------------------------------------------", file=f)
            # print(f"EPSILON: {epsilon:.3f}")
            # print(f"ALPHA: {alpha:.3f}")
        else:
            greedy = False
            # print(
            #     f"-------------- Episode:{episode} --------------")
        states = [[], [], []]
        actions = [[], [], []]
        rewards = [0, 0, 0]

        old_states = env.reset(greedy)
        old_actions = []

        learning_complete = False
        step = 0

        if greedy:
            print(f"学習率: {alpha:.5f}", file=f)
            print(f"ε: {epsilon:.5f}", file=f)

        # 各ステップ
        while True:
            if greedy:
                print(f"\n------- Step:{step} -------", file=f)

            # 各エージェント
            for index, agent in enumerate(env.agents):

                # 終了条件確認
                food_done = env.check_food_run_out()
                all_agents_done = env.check_agents_food_done()
                exceed_max_step = step == lp.MAX_STEPS - 1

                episode_terminal = food_done or all_agents_done or exceed_max_step

                if greedy:
                    print(f"\n本部の在庫: {env.stock}", file=f)
                    print(f"{agent.name}の在庫:{agent.stock}", file=f)
                    diff = agent.old_env_stock - env.stock
                    print(f"{agent.name}視点の本部在庫の変化:{diff}", file=f)
                    if episode_terminal:
                        print("*** 終了条件を満たしています ***", file=f)

                # 状態を観測
                state = agent.observe_state(env.stock, episode_terminal)
                states[index] = state

                if greedy:
                    agent.print_state(state)

                # 行動を決定
                if not episode_terminal:
                    action = agent.decide_action(
                        state, env.stock, greedy, epsilon)
                    actions[index] = action

                    if greedy:
                        if action == es.NUM_FOODS:
                            action_string = "何もしない"
                        else:
                            action_string = f"食品{action}"

                        print(
                            f"{agent.name}のQ値:{agent.brain.Q[state]} 選択した行動: {action_string}", file=f)

                    # 行動をとる
                    if action != es.NUM_FOODS:
                        # エージェントが食品を1つとる
                        agent.grab_food(action)
                        # 本部の在庫が1つ減る
                        env.stock[action] -= 1

                reward = env.get_reward(agent, episode_terminal, greedy)

                if old_states is not None:
                    agent.learn(
                        old_states[index], old_actions[index], reward, state, alpha, greedy)

                if episode_terminal:
                    if greedy:
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

        if epsilon > lp.MINIMUM_EPSILON:
            epsilon = epsilon - lp.EPSILON_DELTA
        if alpha > lp.MINIMUM_ALPHA:
            alpha = alpha - lp.ALPHA_DELTA

        if greedy:
            print(
                f"************ Episode{episode}の結果 ************", file=f)
            print(f"要したステップ数: {step}", file=f)
            print(f"食品の残り個数: {np.sum(env.stock)}", file=f)
            violations = []
            for agent, reward in zip(env.agents, rewards):
                print(
                    f"{agent.name}の在庫: {agent.stock}  制約違反数: {agent.violation}  報酬: {reward}", file=f)
                violations.append(agent.violation)

            ave = np.average(violations)
            st_dev = np.std(violations)

            print(f"制約違反数の平均: {ave:.3f}  分散: {st_dev:.3f}", file=f)
            # print(f"現在のエピソード: {episode}")
            # result_reward_x.append(episode)
            # result_reward_y.append(rewards[0])
            # result_optimal_reward_x.append(episode)
            # result_optimal_reward_y.append(optimal_reward)
            # lines1.set_data(result_reward_x, result_reward_y)
            result_agent_x.append(episode)
            result_ave.append(ave)
            result_dev.append(st_dev)

            for i, agent, line, result in zip(range(es.AGENTS_COUNT), env.agents, lines, results_agents_y):
                # result.append(rewards[i])
                result.append(violations[i])
                line.set_data(result_agent_x, result)

            line_ave.set_data(result_agent_x, result_ave)
            line_dev.set_data(result_agent_x, result_dev)

            # lines1.set_data(result_agents_x, result_agent1_y)
            # lines2.set_data(result_agents_x, result_agent2_y)
            # lines3.set_data(result_agents_x, result_agent3_y)

            ax1.relim()
            ax1.autoscale_view()

            ax2.relim()
            ax2.autoscale_view()

            ax3.relim()
            ax3.autoscale_view()

            # lines1.set_data(result_satisfaction_x, result_satisfaction_y)

            # lines2.set_data(result_optimal_reward_x, result_optimal_reward_y)
            plt.pause(0.001)

    # print("\n---- 最適解 ----")
    # for agent, loss, stock, request in zip(env.agents, optimal_loss_values, optimal_agent_stock, REQUESTS):
    #     diff = stock - request
    #     print(f"{agent.name}: 損失{loss:.1f} 要求との差{diff} 在庫{stock}")
    # print(f"報酬: {optimal_reward:.4f}")
    # print(f"発見したエピソード: {optimal_episode}")

    end_time = datetime.datetime.now()
    print("\n終了時刻: {0:%Y/%m/%d %H:%M:%S}".format(end_time), file=f)

    figure_file_name = f"figure_{file_name_time}.png"
    path = os.path.join(DIR_PATH, "figures", figure_file_name)
    plt.savefig(path)

    print("\n\n終了")

    plt.show()

    f.close()


if __name__ == "__main__":
    run()
