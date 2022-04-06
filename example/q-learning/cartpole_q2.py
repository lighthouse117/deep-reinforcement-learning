import matplotlib.pyplot as plt

# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

import gym
import numpy as np
import random
import math


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))


# Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v0')

# Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)
# Number of discrete actions
NUM_ACTIONS = env.action_space.n
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)


# Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

# Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

# Defining the simulation related constants
NUM_EPISODES = 2000
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199


def simulate():

    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state = state_to_bucket(obv)
        action = select_action(state, explore_rate)

        for t in range(MAX_T):
            # Execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            next_state = state_to_bucket(obv)
            next_action = select_action(next_state, explore_rate)

            # Update the Q based on the result
            next_q = q_table[next_state + (next_action,)]
            q_table[state + (action,)] += learning_rate * (reward +
                                                           discount_factor * (next_q) - q_table[state + (action,)])

            # Setting up for the next iteration
            state = next_state
            action = next_action

            if done:
                print("Episode %d finished after %f time steps" % (episode, t))
                if (t >= SOLVED_T):
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t + 1) / 25)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


simulate()

obs_current = env.reset()
state = state_to_bucket(obs_current)
frames = []
for i in range(1000):
    frames.append(env.render(mode='rgb_array'))
    action = select_action(state, MIN_EXPLORE_RATE)
    obs_next, reward, done, info = env.step(action)
    if done:
        print(i)
        break
    state = state_to_bucket(obs_next)
env.render()
display_frames_as_gif(frames)
