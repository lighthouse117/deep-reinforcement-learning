import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import gym

from agent import Agent

# 最大のステップ数
MAX_STEPS = 200
# 最大の試行回数
NUM_EPISODES = 1000


class Environment():
    def __init__(self, toy_env):
        # 環境を生成
        self.env = gym.make(toy_env)
        # 状態数を取得
        num_states = self.env.observation_space.shape[0]
        # 行動数を取得
        num_actions = self.env.action_space.n
        # Agentを生成
        self.agent = Agent(num_states, num_actions)

    def run(self):
        complete_episodes = 0  # 成功数
        step_list = []
        is_episode_final = False  # 最後の試行
        frames = []  # 画像を保存する変数

        # 試行数分繰り返す
        for episode in range(NUM_EPISODES):
            print(episode)
            observation = self.env.reset()  # 環境の初期化
            for step in range(MAX_STEPS):
                # 最後の試行のみ画像を保存する。
                if is_episode_final:
                    frames.append(self.env.render(mode='rgb_array'))

                # 行動を求める
                action = self.agent.get_action(observation, episode)
                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                observation_next, _, done, _ = self.env.step(action)

                # 報酬を与える
                if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                    if step < 195:
                        reward = -1  # 失敗したので-1の報酬を与える
                        complete_episodes = 0  # 成功数をリセット
                    else:
                        reward = 1  # 成功したので+1の報酬を与える
                        complete_episodes += 1  # 連続成功記録を更新
                else:
                    reward = 0
                # Qテーブルを更新する
                self.agent.update_Q_function(
                    observation, action, reward, observation_next)
                # 観測の更新
                observation = observation_next

                # 終了時の処理
                if done:
                    step_list.append(step + 1)
                    break

            if is_episode_final:
                es = np.arange(0, len(step_list))
                plt.plot(es, step_list)
                plt.savefig("cartpole.png")
                plt.figure()
                patch = plt.imshow(frames[0])
                plt.axis('off')

                def animate(i):
                    patch.set_data(frames[i])

                anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                               interval=50)
                anim.save('movie_cartpole_v0.mp4', "ffmpeg")
                break

            # 10連続成功したら最後の試行を行う
            if complete_episodes >= 100:
                print('100回連続成功')
                is_episode_final = True
