from tensorflow.keras.optimizers import Adam
import random
import copy
import json
import timeit
import warnings
from tempfile import mkdtemp
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from rl.agents.ddpg import DDPGAgent
#from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SARSAAgent
from rl.callbacks import Callback, FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
#from rl.policy import EpsGreedyQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.backend import cast
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     Flatten, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import sys

from fuzzy_controller import *
from enviroment import *
from model import *
from policy import *
from callback import *
from fuzzy_controller import *
import os
from config import *
from MyGlobal import MyGlobals

from rl.agents.dqn import DQNAgent

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def Run_Random(folder_name):
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("Random")
    env.seed(123)
    observation = None
    done = False
    actions = [0, 1, 2, 3, 4, 5]
    for i in range(100000):
        if observation is None:
            try:
                env.reset()
            except Exception as e:
                print(e)
        # Determine the percentage of offload to server
        action = random.choices(actions, weights=(4, 1, 1, 1, 1, 1))[0]
        observation, r, done, info = env.step(action)
        if done:
            done = False
            try:
                env.reset()
            except Exception as e:
                print(e)
            

#using for DQL
def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)

    x = Dense(32, activation='relu')(x)

    x = Dense(32, activation='relu')(x)
  
    x = Dense(16, activation='relu')(x)

    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    return model

def Run_DQL(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    num_actions = NUM_ACTION
    policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("DQL")
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy,gamma=0.9,memory_interval=1)
    #files = open("testDQL.csv","w")
    #files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("DQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DQL.h5f",interval=50000)
    #callback3 = TestLogger11(files)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= 80000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes = 20)
    except Exception as e:
        print(e)
        
def Run_DDQL(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    num_actions = NUM_ACTION
    policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("DDQL")
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy,gamma=0.9,memory_interval=1, enable_double_dqn = True)
    callbacks = CustomerTrainEpisodeLogger("DDQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DDQL.h5f",interval=50000)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= 80000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes = 20)
    except Exception as e:
        print(e)
        
def Run_DuelingDQL(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    num_actions = NUM_ACTION
    policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("DDQL")
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy,gamma=0.9,memory_interval=1, enable_dueling_network = True)
    callbacks = CustomerTrainEpisodeLogger("DuelDQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DuelDQL.h5f",interval=50000)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= 80000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes = 20)
    except Exception as e:
        print(e)
        
def Run_DoubleDuelingDQL(folder_name):
    model=build_model(NUM_STATE, NUM_ACTION)
    num_actions = NUM_ACTION
    policy = EpsGreedyQPolicy(0.1)
    MyGlobals.folder_name = folder_name + '/'
    env = BusEnv("DDQL")
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy,gamma=0.9,memory_interval=1, enable_double_dqn = True,
              enable_dueling_network = True)
    callbacks = CustomerTrainEpisodeLogger("DDuelDQL_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_DDuelDQL.h5f",interval=50000)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= 80000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
        dqn.policy = EpsGreedyQPolicy(0.0)
        dqn.test(env, nb_episodes = 20)
    except Exception as e:
        print(e)

def Run_FDQO():
    FDQO_method = Model_Deep_Q_Learning(14,4)
    model = FDQO_method.build_model()
    #Create enviroment FDQO
    env = BusEnv("FDQO")
    env.seed(123)
    #create memory
    memory = SequentialMemory(limit=5000, window_length=1)
    #create policy 
    policy = EpsGreedyQPolicy(0.0)
    #open files
    files = open("testFDQO.csv","w")
    files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("FDQO_5phut.csv")
    callback2 = ModelIntervalCheckpoint("weight_FDQO.h5f",interval=50000)
    callback3 = TestLogger11(files)
    model.compile(Adam(lr=1e-3), metrics=['mae'])
    model.fit(env, nb_steps= 104838, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    files.close()

if __name__=="__main__":
    # for i in range(1, 6):
    #     Run_DQL("DQN" + str(i))
    #Run_DQL("DQN1")
    #Run_DDQL("DDQN3_no_energy")
    Run_DuelingDQL("DuelingDQN3_"+str(NUM_ACTION - 1)+"VS")
    #Run_DoubleDuelingDQL("DoubleDuelingDQN1")
    #Run_Random("Random_4_1_1_1_1_1")

















