# script that has the random number check and mobilization check function
import plotly.graph_objects as go
import numpy as np
import time

#np.random.seed(time.time_ns() % 10)
t = int( time.time() * 1000.0 )
np.random.seed( ((t & 0xff000000) >> 24) +
             ((t & 0x00ff0000) >>  8) +
             ((t & 0x0000ff00) <<  8) +
             ((t & 0x000000ff) << 24)   )

def mob_check(thresh):
    d100_roll = np.random.rand(1)
    #print (d100_roll)
    if (d100_roll < thresh):
         return True
    else:
        return False

def matchNames(x,r):
    returnIdx = -1
    for idxX, rost in enumerate(r):
        if rost[0] == x[0]:
            returnIdx = idxX
    return(returnIdx)

def sample_feature_matrix():
    x = np.random.rand(1000)
    print (x)

def select_from_roster(r):
    count = -1;
    tmp = []

    for n in r:
        if n[1] == "unassigned":
            tmp.append(n)
            count +=1

    if count > -1:
        return(tmp[np.random.randint(0,count+1)])
    else:
        return([])
