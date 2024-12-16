import deepq
import gym
import carla_env
import time

def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    env = gym.make('CarlaEnv-pixel-town01-v1')
    deepq.learn(env, model_identifier=str(time.time()))
    env.close()


if __name__ == '__main__':
    main()

