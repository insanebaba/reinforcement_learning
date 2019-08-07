import numpy as np
from os import system
from time import sleep
import random

class QLearningModel(object):
    """ A Q learning model """

    def __init__(self, env):
        # Environment variabless
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        #Q Table
        self.qTable = np.zeros(
            [self.env.observation_space.n, self.env.action_space.n])
        self.penalties = []
        #Training parameters
        self.epsilon_ExploreExploitFactor = 0.8
        self.alpha_LearningRate = 0.1
        self.gamma_DiscountFactor = 0.6

    def get_Q_table(self):
        return self.qTable

    def update_Q_value(self, reward, state, action, newState):
        """ update the q table with new reward """
        currentReward=self.qTable[state, action]
        self.qTable[state, action] = currentReward + self.alpha_LearningRate * (reward + self.gamma_DiscountFactor * np.max(self.qTable[newState]) - currentReward)

    def epsilon_greedy(self, state):
        """ greedily select an action based on epsilon_ExploreExploitFactor, 
        if random float(0,1) < epsilon  ->  explore , else exploit
        """
        if random.uniform(0, 1) < self.epsilon_ExploreExploitFactor:
            return self.action_space.sample()
            # explore
            # return np.random.randint(0, self.action_space.n)
        else:
            # exploit
            return np.argmax(self.qTable[state])

    def train(self, epochs=5000):
        #Run for these many epochs
        for i in range(1, epochs):
            #Is the episode over(successful drop or fail dropped)
            system("clear")
            print((i%50)*'=','>')#progress bar
            done = False
            #Initialise the env for an episode
            state = self.env.reset()
            #begin the episode

            # penalty = 0
            while not done:
                #Pick an action
                action = self.epsilon_greedy(state)
                #Take an action
                next_state, reward, done, _ = self.env.step(action)
                #Update the Q table
                self.update_Q_value(reward, state, action, next_state)
                state = next_state
                #Specific only to taxi
                # if reward == -10:
                    # penalty += 1
            
                #update the epsilon
            if(i % 300 == 0):
                self.epsilon_ExploreExploitFactor *= self.epsilon_ExploreExploitFactor

    def run_an_episode(self, stupid_strategy=False):
        done = False
        state = self.env.reset()
        timestep=0
        penalty =0
        while not done :
            system("clear")

            ## choose an action
            if stupid_strategy:
                # (this takes random actions)
                action = self.env.action_space.sample()
            else:
                # your agent here 
                print(f'action proposed from Q table:{np.argmax(self.qTable[state])}')
                action = np.argmax(self.qTable[state])

            #Take that action
            
            state, reward, done, _ = self.env.step(action)
            self.env.render()
            if reward == -10:
                penalty += 1
            timestep+=1
            ################################
            
            print(f'state={state}', f'reward={reward}',
                  f'timestep={timestep}',sep='\n')

            timestep += 1
            sleep(1.0)
