import os
from pathlib import Path

LINK_PROJECT = Path(os.path.abspath(__file__))
LINK_PROJECT = LINK_PROJECT.parent.parent
#print(LINK_PROJECT)
DATA_DIR = os.path.join(LINK_PROJECT, "data")
RESULT_DIR = os.path.join(LINK_PROJECT, "result")
DATA_TASK = os.path.join(LINK_PROJECT, "data_task/high_deadline")
COMPUTATIONAL_CAPACITY_900 = 1 # Ghz
COMPUTATIONAL_CAPACITY_901 = 1.2 # Ghz
COMPUTATIONAL_CAPACITY_902 = 1 # Ghz
COMPUTATIONAL_CAPACITY_LOCAL = 3 # Ghz
CHANNEL_BANDWIDTH = 10 # MHz
List_COMPUTATION = [COMPUTATIONAL_CAPACITY_900, COMPUTATIONAL_CAPACITY_901, COMPUTATIONAL_CAPACITY_902, COMPUTATIONAL_CAPACITY_LOCAL]
Pr = 46 # dBm
SIGMASquare = 100 # dBm   background noise power
PATH_LOSS_EXPONENT = 4  # alpha
class Config:
    Pr = 46
    Pr2 = 24
    Wm = 10
    length_hidden_layer=4
    n_unit_in_layer=[16, 32, 32, 8]
