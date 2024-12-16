#Initialize carla in this code and complete the main function
import deepq
import gym
import carla_env

# run python3 ../evaluate_racing.py in /carla_env

def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    env = gym.make('CarlaEnv-pixel-town01-v1')
    deepq.evaluate(env, '../agent.pt')
    env.close()

if __name__ == '__main__':
    main()
