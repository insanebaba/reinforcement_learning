# !pip install cmake 'gym[atari]' scipy
import gym
from os import system
from time import sleep
from q_learning import QLearningModel

system("clear")
# Create environment
################################################################################################
env = gym.make("Taxi-v2")
observation = env.reset()
################################################################################################

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
#------------------------------------------------------------------------------------------------------------------------------------
qModel=QLearningModel(env)


# qModel.run_an_episode(stupid_strategy=True)
# print("training model in 5 seconds")
# sleep(5)

qModel.train(10000)
print("running an episode after Training ")
sleep(5)
qModel.run_an_episode()
# print(qModel.getQTable().shape)
env.close()
