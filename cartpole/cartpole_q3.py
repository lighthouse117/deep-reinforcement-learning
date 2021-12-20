import gym
import numpy as np
import time
import math
from statistics import mean, median
import matplotlib.pyplot as plt


class EnvCartPole:
    """
    CartPole-v0 強化学習クラス
    @see https://github.com/openai/gym/wiki/CartPole-v0
    """

    def __init__(self):
        """
        コンストラクタ
        """
        self.env = gym.make('CartPole-v0')
        # 各種観察値の等差数列を生成
        # pole_velocityの範囲は特に重要
        self.bins = {
            'cart_position': np.linspace(-2.4, 2.4, 3)[1:-1],
            'cart_velocity': np.linspace(-3, 3, 5)[1:-1],
            'pole_angle': np.linspace(-0.5, 0.5, 5)[1:-1],
            'pole_velocity': np.linspace(-2, 2, 5)[1:-1]
        }
        num_state = (
            (len(self.bins['cart_position']) + 1)
            * (len(self.bins['cart_velocity']) + 1)
            * (len(self.bins['pole_angle']) + 1)
            * (len(self.bins['pole_velocity']) + 1)
        )
        self.q_table = np.random.uniform(
            low=-1, high=1, size=(num_state, self.env.action_space.n))
        self.q_table_update = [[math.floor(time.time()), 0]] * num_state
        self.save_q_text()
        self.episodes = []
        self.episode_rewards = []
        self.episode_rewards_mean = []
        self.episode_rewards_median = []

    def save_q_text(self):
        """
        Qテーブルをファイルに保存する
        ・q_table_cartpole.txt: Qテーブル情報（行: 状態、列: 行動）
        ・q_table_cartpole_update.txt: Qテーブル更新履歴（行 状態、列: 更新日時UNIXタイムスタンプ、更新回数）
        :return: void
        """
        np.savetxt('q_table_cartpole.txt', self.q_table)
        np.savetxt('q_table_cartpole_update.txt', self.q_table_update, '%.0f')

    def digitize_state(self, observation):
        """
        状態のデジタル化をする
        :param observation: 観察値
        :return: デジタル化した数値
        """
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        state = (
            np.digitize(cart_position, bins=self.bins['cart_position'])
            + (
                np.digitize(
                    cart_velocity, bins=self.bins['cart_velocity']) * (len(self.bins['cart_position']) + 1)
            )
            + (
                np.digitize(pole_angle, bins=self.bins['pole_angle'])
                * (len(self.bins['cart_position']) + 1)
                * (len(self.bins['cart_velocity']) + 1)
            )
            + (
                np.digitize(pole_velocity, bins=self.bins['pole_velocity'])
                * (len(self.bins['cart_position']) + 1)
                * (len(self.bins['cart_velocity']) + 1)
                * (len(self.bins['pole_angle']) + 1)
            )
        )
        return state

    def get_action(self, state, episode):
        """
        状態から次のアクションを取得する
        （εグリーディ法使用）
        :param state: 状態値
        :param episode: エピソード数
        :return: 次のアクション
        """
        # 重要
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.choice([0, 1])
        return action

    def update_q_table(self, state, action, reward, next_state):
        """
        Qテーブルを更新する
        :param state: 前の状態
        :param action: 前のアクション
        :param reward: 前のアクションにより獲得した報酬
        :param next_state: 次の状態
        :return: void
        """
        alpha = 0.2     # 学習率
        gamma = 0.99    # 割引率
        max_q_value = max(self.q_table[next_state])
        current_q_value = self.q_table[state, action]
        self.q_table[state, action] = (
            current_q_value
            + alpha
            * (reward + (gamma * max_q_value) - current_q_value)
        )
        self.q_table_update[state] = [math.floor(
            time.time()), self.q_table_update[state][1] + 1]

    def run(self, num_episode, num_step_max):
        """
        学習を実行する
        :param num_episode: エピソード数
        :param num_step_max: 最大ステップ数
        :return: True: 課題解決成功、false: 課題解決失敗
        """
        num_solved = 0
        num_solved_max = 0
        self.episodes = []
        self.episode_rewards = []
        self.episode_rewards_mean = []
        self.episode_rewards_median = []

        for episode in range(num_episode):
            observation = self.env.reset()
            state = self.digitize_state(observation)
            action = np.argmax(self.q_table[state])
            episode_reward = 0

            for step in range(num_step_max):
                if episode % 100 == 0:
                    self.env.render()
                observation, reward, done, _ = self.env.step(action)
                # 重要
                if done and step < num_step_max - 1:
                    reward -= num_step_max
                episode_reward += reward

                next_state = self.digitize_state(observation)
                self.update_q_table(state, action, reward, next_state)
                action = self.get_action(next_state, episode)
                state = next_state
                if done:
                    break
            print(f'episode: {episode}, episode_reward: {episode_reward}')
            if episode_reward >= 195:
                num_solved += 1
            else:
                num_solved = 0
            if num_solved_max < num_solved:
                num_solved_max = num_solved
            self.episodes.append(episode)
            self.episode_rewards.append(episode_reward)
            self.episode_rewards_mean.append(mean(self.episode_rewards))
            self.episode_rewards_median.append(median(self.episode_rewards))

        return self.finish(num_solved_max)

    def finish(self, num_solved_max):
        """
        学習を終了する
        :param num_solved_max: 最大連続解決数
        :return: True: 課題解決成功、false: 課題解決失敗
        """
        self.env.close()
        self.save_q_text()
        plt.plot(self.episodes, self.episode_rewards, label='reward')
        plt.plot(self.episodes, self.episode_rewards_mean, label='mean')
        plt.plot(self.episodes, self.episode_rewards_median, label='median')
        plt.legend()
        plt.show()
        print(f'num_solved_max: {num_solved_max}')
        return num_solved_max > 100


if __name__ == '__main__':
    this = EnvCartPole()
    result = this.run(num_episode=2000, num_step_max=200)
    if result:
        print('CLEAR!')
    else:
        print('FAILED...')
