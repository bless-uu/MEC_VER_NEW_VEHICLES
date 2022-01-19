#from code.config import Config, DATA_DIR, RESULT
from config import DATA_DIR
import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import copy
import os

# data900 = pd.read_excel(os.path.join(DATA_DIR, "data9000.xlsx"), index_col=0).to_numpy()
# data900 = data900[:, 13:15]
# print(data900)
for i in range(6):
    print(np.random.randint(1000,2000,10))