import gym
import numpy as np

env = gym.make('Taxi-v3').env
env.reset() #Starting the program with a new randomised environment

q_table = np.zeros([env.observation_space.n, env.action_space.n]) #Creating a Q-table in which each state in the observation space is associated with each and all of the action from the action space

action = env.action_space.sample() #Choosing a random action for agent to take
new_state, reward, is_done, _ = env.step(action) #Agent performing the random action and returned information being stored in proper variables

if is_done:
    env.reset()

env.close()