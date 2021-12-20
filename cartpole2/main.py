from env import Environment

TOY = "CartPole-v0"


def main():
    cartpole = Environment(TOY)
    cartpole.run()


if __name__ == "__main__":
    main()
