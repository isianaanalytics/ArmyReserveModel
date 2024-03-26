#example which will predict, when given which senior leaders are able to mobilize,
# if the command group is adequately filled
# Virgilio Villacorta, 17 March 2024

#first, generate adjaceny matrix. add identity and normalize it
#d_mod (normalizing matrix: count number of connections to node and put in diagonal)

#next, generate feature vectors and training_outputs
#feature vector: 1xN, where N is number of senior leaders available from BDE and Bn
# value is 1 if mobilizing, and 0 if not mobilizing
import plotly.graph_objects as go
import numpy as np
from scipy.linalg import sqrtm
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import time
from assign_o4s import assign_O4s_one_run
from assign_o5s import assign_O5s_one_run
from assign_E9s import assign_E9s_one_run
from assign_tools import *
from neuralnet_backprop import NeuralNetwork

def adjaceny_matrix():
    #connect all LTCs to each other, both CSMs, and all majors
    #note this is a directional connection, so the matrix is non-symmetric
    #this is because vacancies are only filled one way (into the mobilizing CSSB)
    #those outside of the CSSB are in nonmobilizing units, so aren't a prior to
    #fill their vacancies
    #also note we add the identity matrix for self-connections, since those sitting
    #in the position are first priority to fill the posiiton
    #this function returns a normalized version of adjaceny matrix

    A =np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
         [0, 0, 0, 1, 0, 1, 1, 0, 0, 1]])

    D = np.zeros_like(A)
    np.fill_diagonal(D, np.asarray(A.sum(axis=1)).flatten())
    D_invroot = np.linalg.inv(sqrtm(D))
    A_hat = D_invroot @ A @ D_invroot

    return (A_hat)

def position_labels(label):
    match label:
        case 'unassigned':
            return(0)
        case 'nondeployable':
            return (0)
        case 'BC':
            return (1)
        case 'CSM':
            return (2)
        case 'XO':
            return (3)
        case 'SPO':
            return (4)

def read_feature_vect(input):
    out_phrase = []
    if input[0] == 1:
        out_phrase.append("LTC A is BC, ")
    elif input[1] == 1:
        out_phrase.append("LTC B is BC, ")
    elif input[2] == 1:
        out_phrase.append("LTC C is BC, ")
    else:
        out_phrase.append("BC unfilled, ")

    if input[3] == 2:
        out_phrase.append("CSM A is CSM, ")
    elif input[4] == 2:
        out_phrase.append("SGM B is CSM, ")
    else:
        out_phrase.append("CSM unfilled, ")

    if input[5] == 3:
        out_phrase.append("MAJ A is XO, ")
    elif input[6] == 3:
        out_phrase.append("MAJ B is XO, ")
    elif input[7] == 3:
        out_phrase.append("MAJ C is XO, ")
    elif input[8] ==3:
        out_phrase.append("MAJ D is XO, ")
    elif input[9] ==3:
        out_phrase.append("MAJ E is XO, ")
    else:
        out_phrase.append("XO unfilled, ")

    if input[5] == 4:
        out_phrase.append("MAJ A is SPO. ")
    elif input[6] == 4:
        out_phrase.append("MAJ B is SPO. ")
    elif input[7] == 4:
        out_phrase.append("MAJ C is SPO. ")
    elif input[8] ==4:
        out_phrase.append("MAJ D is SPO. ")
    elif input[9] ==4:
        out_phrase.append("MAJ E is SPO. ")
    else:
        out_phrase.append("SPO unfilled, ")

    return (out_phrase)

def read_prediction(value, thr):
    if value > thr:
        return ("prediction: ", f'{value:.4f}', " Command Group filled.")
    else:
        return ("prediction: ", f'{value:.4f}', " Command Group unfilled.")

def feature_vetor(rost5,rost9,rost4):
    #generate feature vector
    out = []
    for tmp1 in rost5:
        out.append(position_labels(tmp1[1]))
    for tmp1 in rost9:
        out.append(position_labels(tmp1[1]))
    for tmp1 in rost4:
        out.append(position_labels(tmp1[1]))
    return (np.array(out))

def is_cmd_grp_filled(rost5, rost9, rost4):
    BC = 0
    for x in rost5:
        try:
            x.index('BC')
            BC = 1
        except:
            BC +=0

    CSM = 0
    for x in rost9:
        try:
            x.index('CSM')
            CSM = 1
        except:
            CSM +=0

    XO = 0
    for x in rost4:
        try:
            x.index('XO')
            XO = 1
        except:
            XO +=0

    if BC+CSM+XO >=2:
        return (1)
    else:
        return (0)


#initialize neural network
nn = NeuralNetwork([10, 6, 1], alpha=0.5)
Adj = adjaceny_matrix()

forig = np.zeros(10)

#range of 1000 to generate 1000 batches of runs of the simulation
# since batch size is 10, number of actual runs is 1000 *10 or 10,000
for eps in range (500):
    fmatrix = np.zeros(10)

    outvect = np.zeros(1)
    print ("epoch: ", eps)
#range of 10 to batch size
    for n in range (10):
        rostO5s = assign_O5s_one_run(0.4)
        rostE9s = assign_E9s_one_run(0.4)
        rostO4s= assign_O4s_one_run(0.3)

        fvect = feature_vetor(rostO5s,rostE9s,rostO4s)
        #nn_input = np.array(Adj @ fvect.T)

        #now, make training output
        out = np.array([is_cmd_grp_filled(rostO5s, rostE9s, rostO4s)])

        fmatrix = (np.vstack((fmatrix,fvect)))
        outvect = np.append(outvect, out)


        #print (fmatrix)
    fmatrix = np.delete(fmatrix,(0), axis = 0)
    outvect = np.delete (outvect,(0))

    features = np.array (fmatrix)

    output = np.array(outvect)
    nn.fit(features, output, Adj)

    if (eps % 2 == 0):
        forig = features
#features_test = np.array([
#    [1, 0, 0, 2, 0, 3, 4, 0, 0, 0],
#    [0, 1, 0, 2, 0, 0, 4, 0, 3, 0],
#    [0, 0, 0, 0, 0, 3, 0, 0, 4, 0],
#    [0, 1, 0, 0, 0, 0, 0, 0, 0, 4]])
#output_test = np.array([
#    [1], [1], [0], [0]])

for (x, target, z) in zip(features, output, forig):
	# make a prediction on the data point and display the result
	# to our console

    pred = nn.predict(x)[0][0]
    predz = nn.predict(z)[0][0]    #print (x, z)
    step = 1 if pred > 0.5 else 0
    #print("[INFO] data={}, ground-truth={}, pred={:.4f}".format(
	#	x, target, pred))
    print (x, read_feature_vect(x), read_prediction(pred, 0.5))
    print (z, read_feature_vect(z), read_prediction(predz, 0.5))
