class LearningParameters:
    MAX_EPISODES = 1000001
    MAX_STEPS = 100
    GREEDY_CYCLE = 1000

    GAMMA = 0.98

    INITIAL_EPSILON = 1.0
    MINIMUM_EPSILON = 0.1
    EPSILON_DELTA = (INITIAL_EPSILON - MINIMUM_EPSILON) / (MAX_EPISODES * 0.95)

    INITIAL_ALPHA = 0.5
    MINIMUM_ALPHA = 0.0001
    ALPHA_DELTA = (INITIAL_ALPHA - MINIMUM_ALPHA) / (MAX_EPISODES * 0.95)


class EnvironmentSettings:
    AGENTS_COUNT = 3
    FOODS = [20, 20, 20]
    NUM_FOODS = len(FOODS)
    REQUESTS = [
        [10, 10, 10],
        [5, 10, 5],
        [5, 5, 10],
    ]
