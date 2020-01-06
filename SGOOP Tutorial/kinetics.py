import numpy as np

def persistence(labels):
    # converts trajectory of cluster labels to states and time spent per state
    # in the format [label,time spent]
    states = []
    current = [labels[0],0]
    for label in labels:
        if label == current[0]:
            current[1]+=1
        else:
            states.append(current)
            current = [label,1]
    states.append(current)
    return np.array(states)


def commitment(states,t_commit):
    # computes which state visits last a time greater than or equal to t_commit
    # input and output states are in the format [label,time spent]
    metastable = states[states[:,0]>-1]
    committed = metastable[metastable[:,1]>=t_commit]
    return committed


def count_path(states,path):
    # counts the number of paths taken matching the given path
    # states is in the format [label,time] and path is vector with a specified
    # order of states.
    # the output is an integer count of paths
    edges = np.shape(path)[0]
    routes = np.zeros([np.shape(states)[0]-edges+1,edges])
    for i in range(np.shape(routes)[0]):
        routes[i,:] = np.array([states[i:i+edges,0]])
    return np.shape(routes[(routes == tuple(path)).all(axis=1)])[0]


def find_wells(prob):

    energy = []
    for i in (range(len(prob))):
        if prob[i] == 0:
            energy.append(np.inf)
        else:
            energy.append(-1 * np.log(prob[i]))

    wells = 0
    max = np.inf
    min = np.inf
    d = 1
    i = 0
    for x in energy:
        if x > max:
            max = x
            if (max - min > 1):
                min = x
                d = 1
        elif x < min:
            min = x
            if (max - min > 1):
                if d == 1:
                    wells = wells + 1
                max = x
                d = -1
        i = i + 1

    return wells


def find_barriers(prob):
    # finds the barriers > 1 kT for a given probability distribution
    # returns barrier indices
    energy = []
    for i in (range(len(prob))):
        if prob[i] == 0:
            energy.append(np.inf)
        else:
            energy.append(-1 * np.log(prob[i]))

    barriers=[]
    max = np.inf
    min = np.inf
    d = 1
    i = 0
    bar = [i,max]
    for x in energy:
        if x > max:
            max = x
            if max > bar[1]:
                bar = [i,max]
            if (max - min > 1):
                min = x
                d = 1
        elif x < min:
            min = x
            if (max - min > 1):
                if d == 1:
                    barriers.append(bar[0])
                    bar = [i,0]
                max = x
                d = -1
        i = i + 1

    return barriers[1:]

