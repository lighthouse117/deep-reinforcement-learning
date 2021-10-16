# 深層強化学習 DDQN(Double Deep Q-Network)のテスト
# Q値を評価するターゲットネットワークと、行動決定を行うQ-ネットワークの2つがある
# 適用するタスク：OpenAI Gym CartPole-v0
# 参考サイト：PyTorch公式チュートリアル "Reinforcement Learning (DQN) Tutorial"

# Tomoshi Iiyama
# 2021-10-14


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# matplotlibの設定
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPUを使用する場合
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1つの状態遷移を表す名前付きタプル
# （現在の状態, 行動, 次の状態, 報酬）
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    """
    Experience Replay(経験再生)を実装するクラス
    学習データを一定数メモリに保存し、ランダムにサンプリングすることで
    時系列の相関をなくすことができる
    """

    def __init__(self, capacity):
        """メモリを初期化"""
        # 状態遷移を保持するバッファ
        self.memory = deque([], maxlen=capacity)
        # deque: double-ended queue （両端キュー）
        # 両サイドからデータを取り出したり追加できる
        # リストに比べて先頭・末尾の操作がO(1)の計算量で済む
        # 最大要素数のcapacityを超えると古い要素から削除される

    def push(self, *args):
        """状態遷移をメモリに保存"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """指定した数の状態遷移をメモリからランダムに取り出す"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    Q関数を畳み込みニューラルネットワーク(CNN)で近似するクラス
    ある状態を入力すると2つの値が出力される
    （2種類の行動をそれぞれ選択したときのリターン）
    """

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # 線形レイヤの入力次元数はconv2dレイヤーの出力に依存するため、
        # 入力画像サイズをもとにconv2dレイヤーの出力サイズを計算
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        """
        最適化中、次の行動決定のための要素、もしくはバッチによって呼ばれる
        tensor([[left0exp,right0exp]...])を返す
        """
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


"""
レンダリングした画像を抽出して処理するための関数たち
"""
# リサイズする関数（変数ではない）
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    """
    カートの位置を計算する
    """
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # カートの真ん中


def get_screen():
    """
    スクリーンを生成
    """
    # gymで使用されるスクリーンサイズ(400x600x3)より、
    # 実際のスクリーンが大きい場合がある。(800x1200x3など)
    # 画像データ形式をPyTorch標準のCHW（色, 高さ, 幅）に並び変える
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # スクリーンの下半分にカートがあるので、上下をトリミング
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # ふちをトリミングしてカートが中心の四角い画像にする
    screen = screen[:, :, slice_range]
    # floatに変換して、再拡大したあと、torchのtensorに変換
    # (変数のコピーは不要)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # リサイズ後、バッチの次元を追加してBCHWという形式にする（バッチ, 色, 高さ, 幅）
    return resize(screen).unsqueeze(0)


"""
描画のテスト
"""
env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()


BATCH_SIZE = 128    # バッチサイズ
GAMMA = 0.999       # γ（時間割引率）
EPS_START = 0.9     # 開始時のε（ランダム行動の選択確率）ε-greedy法
EPS_END = 0.05      # 終了時のε（これに向かって指数関数的に減衰させる）
EPS_DECAY = 200     # εの減少率
TARGET_UPDATE = 10  # ターゲットネットワークを更新するエピソード間隔

# レイヤーを正しく初期化するために、スクリーンサイズを取得
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# 選択できる行動の数をgymから取得（今回は右か左の2つ）
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    """
    ある状態において行動を選択する
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    """
    結果をプロットする
    """
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100エピソードの平均をとってプロット
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # プロットを更新するため一時停止
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    """
    モデルをトレーニングする
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # バッチを転置
    # Transitionのバッチ配列 → バッチ配列のTransition
    batch = Transition(*zip(*transitions))

    # 最終状態以外の状態を取り出して、バッチの要素を連結
    # (最終状態になるのはシミュレーションが終了した後)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a)を計算する
    # モデルでQ(s_t)を算出して2つの値が出力された後、
    # 実際に取られた行動のほうのQ値を取得
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # max{Q(s_t+1, a)}を求める
    # 全ての遷移先の状態において、最大のQ値を計算
    # 最終状態の場合は0
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # 遷移先状態のQ値の期待値を計算 (目標値)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 遷移先状態の現在のQ値と目標値からフーバー損失関数を計算
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # ネットワークの更新
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 100
for i_episode in range(num_episodes):
    # 環境と状態を初期化
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # 行動を選択して出力
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 遷移先状態を取得
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 状態遷移をメモリに保存
        memory.push(state, action, next_state, reward)

        # 次の状態へ遷移
        state = next_state

        # モデルを最適化（Qネットワーク）
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    # Qネットワークの重みとバイアスをコピーして、ターゲットネットワークを更新
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
