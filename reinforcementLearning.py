import gym
import numpy as np
import pandas as pd
import random

env = gym.make('Taxi-v3').env
env.reset() #Starting the program with a new randomised environment

q_table = np.zeros([env.observation_space.n, env.action_space.n]) #Creating a Q-table in which each state in the observation space is associated with each and all of the action from the action space

alpha = 0.1 #Learning rate
gamma = 0.6 #Discount factor: Long-term vs short-term reward relevance
epsilon = 0.1 #Exploration/Explotation ratio

num_ephocs = 1000000 #Number of ephocs the agent should complete

while num_ephocs > 0:

    current_state = env.s

    if random.random() < epsilon: #Exploration phase
        action = env.action_space.sample() #Choosing a random action for agent to take
    else: #Explotation phase
        action = np.argmax(q_table[current_state]) #Search the row corresponding to the current state for the action that returns highest reward
    
    new_state, reward, is_done, _ = env.step(action) #Agent performing the chosen action and the returned information being stored in proper variables
    q_table[current_state, action] = ((1 - alpha) * q_table[current_state, action]) + (alpha * (reward + (gamma * np.max(q_table[new_state, :])))) #Update Q-table
    
    if is_done: #Variable set to true when the action taken lead to a successful passenger drop-off
        num_ephocs -= 1 #Update the ephoc counter variable
        env.reset() #Reset the environment to a new random state

q_table = q_table.ravel() #Flatten the Q-table

#The following block prepares the data format to be submitted as required in a .csv file
output_data = {'Id': np.arange(1, len(q_table) + 1, 1), 'Value': [q_value for q_value in q_table]} 
output_data = pd.DataFrame(data = output_data)
output_data.to_csv('QTable_Values.csv', index = False)

env.close() #Close the environment to avoid leaking data