import plotly.graph_objects as go
import numpy as np
import time

from assign_tools import *

# This will run 10k simulations of mobilization manning just for the two major positions
# in the Battalion.  Both positions are currently filled, and there are 2 other majors
# from higher--brigade level--available if needed to fill in for one or both of
# the two battalion positions if they become vacant

#for one run of the simulation, check eligibility for mobiizing officer.
# if pass, that position remains filled.
# if fails, select randomly one of the two available officers from brigade,
# if that person passes, fills vacancy.  if not, select other person
# if both people cannot fill vacancy, coin flip if check other battalion member should fill vacancy
# *** prob to pass mob check first time is 0.9; if passes, that increases to 0.98

#roster is a list of people, identified by
#  (name, position, mob readiness (0.0 to 1.0 probability), months in position)

def reset_mp(mp):
    mp = [("BC", False)]
    return (mp)

def reset_r(r, thresh):
    r = [
        ("LTC A", "BC", thresh, 18),
        ("LTC B", "unassigned", thresh, 0),
        ("LTC C", "unassigned", thresh, 0)
        ]
    return (r)

def assign_O5s_one_run(threshold):
    roster = []
    mob_positions = []


#first pass is to check if all SMs currently assigned are qualified to mobilize
    mob_positions = reset_mp(mob_positions)
    roster = reset_r(roster, threshold)

    for idxM, x in enumerate(mob_positions):
        xList = list(x)

        for idxSM, SM in enumerate(roster):
            if SM[1] == x[0]:
                if mob_check(SM[2]):
                    newSM = list(SM)
                    newSM[2] = 0.99
                    roster[idxSM] = newSM

                    xList[1] = True
                    mob_positions[idxM] = xList
                else:
                    newSM = list(SM)
                    newSM[1] = 'nondeployable'
                    newSM[2] = 0.0
                    roster[idxSM] = newSM

#print (roster)
#next pass, fill up vacancies
    for idxM, x in enumerate(mob_positions):
        if (not(x[1])):
            stopWhile = False

            while ( not(stopWhile)):
                xList = list(x)
                SM = select_from_roster(roster)
                if SM == []:
                    stopWhile = True

                else:
                    newSM = list(SM)
                    idx = matchNames(SM, roster)

                    if mob_check(SM[2]):
                        newSM[1] = x[0]
                        newSM[2] = 0.99
                        xList[1] = True
                        mob_positions[idxM] = xList
                        stopWhile = True

                    else:
                        newSM[1] = 'nondeployable'
                        newSM[2] = 0.0
                roster[idx] = newSM

    #print(roster)
    #print (mob_positions)
    #print()
    return (roster)


#print (filled_names)
