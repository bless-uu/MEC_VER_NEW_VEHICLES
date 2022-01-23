import random as rd
import numpy as np
from pathlib import Path
import os
from config import DATA_LOCATION
path =os.path.abspath(__file__)
path =Path(path).parent.parent
for i in range(100):
    with open("{}/{}/datatask{}.csv".format(str(path),DATA_LOCATION,i),"w") as output:
        # indexs=rd.randint(900,1200)
        # m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
        # m1 = np.random.randint(1000,2000,indexs) # p in
        # m2 = np.random.randint(100,200,indexs) # p out
        # m3 = np.random.randint(500,1500,indexs) #Computational resource
        # m4 = 1+np.random.rand(indexs)*2 #deadline
        
        indexs=1000
        m = np.sort(np.random.randint(i*120,(i+1)*120,indexs))
        m1 = np.random.randint(100,200,indexs)/1000 # p in Mb
        m2 = np.random.randint(10,11,indexs)/1000 # p out Mb
        m3 = np.random.randint(500,800,indexs)/1000 #Computational resource GHz
        m4 = 1+np.random.rand(indexs) #deadline
        # indexs=rd.randint(1000,1000)
        # m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
        # m1 = np.random.randint(1000,1200,indexs)
        # m2 = np.random.randint(100,110,indexs)
        # m3 = np.random.randint(2,3,indexs)
        # m4 = 5+np.random.rand(indexs)*2
        
        for j in range(indexs):
            output.write("{},{},{},{},{}\n".format(m[j],m3[j],m1[j],m2[j],m4[j]))
    #import pdb;pdb.set_trace()