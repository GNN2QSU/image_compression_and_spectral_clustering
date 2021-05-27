import pandas as pd
import numpy as np
from time import process_time


def construct_distance_matrix(X, points, dist_type='l2'):
    '''
    computes distance matrix of datapoints in X to multiple points
    Args:
        X: m x d array where m = # datapoints, d = dim
        points: n x d ndarray
        dist_type: type of distance, 'l2' or 'l1'
    Returns:
        distance matrix D: m x n ndarray where m = # datapoints in X, n = number of points to compute distances from
            
    D[i, j] = distance of point i in intialization array to point j in the argument points, where:
        i = {0, 1, ..., m}
        j = {0, 1, ..., n}
    '''
    # initialize empty distance matrix
    m = X.shape[0]
    d = X.shape[1]
    points = points.reshape((-1, d))
    n = points.shape[0]
    D = np.empty((m, n))
    
    if dist_type == 'l2':
        def dist_func(X, y):
            return np.linalg.norm(X-y, axis=1)
    
    if dist_type == 'l1':
        def dist_func(X, y):
            return (np.abs(X-y)).sum(axis=1)
        
    # compute distance and assign to D
    for j in range(n):
        D[:, j] = dist_func(X, points[j, :])    
        
    return D


def run_step_A(X, centroids, dist_type='l2'):
    '''
    run step A of KMeans clustering: assignment of clusters based on centroids
    Args: 
        X: m x d array where m = # datapoints, d = dim
        centroids: k x d array where k = # of clusters
    Returns:
        assigned_cluster: 0 x m array
    '''
    D = construct_distance_matrix(X, centroids, dist_type=dist_type)
    cluster = np.argmin(D, axis=1)
    return cluster


def run_step_B(X, k, cluster, initial_centroids, dist_type='l2'):
    '''
    run step B of KMeans clustering: computation of centroids based on assigned clusters
    Args:
        X: m x d array where m = # datapoints, d = dim
        k: # of clusters
        cluster: 0 x m array
    Returns:
        centroids: k x d array
    '''
    centroids = []
    for i in range(k):
        X_cluster = X[cluster == i, :]
        if len(X_cluster) > 0:
            if dist_type == 'l2':
                centroids.append(X_cluster.mean(axis=0).reshape((1, -1)))   # adds additional dim for concatenation
            if dist_type == 'l1':
                centroids.append(np.median(X_cluster, axis=0).reshape((1, -1)))   # adds additional dim for concatenation
        else:
            centroids.append(initial_centroids[i, :].reshape((1, -1)))
    centroids = np.concatenate(centroids, axis=0)
    return centroids


def initialize_centroids(X, K, dist_type='l2', init_method='plusplus', rng=None):
    '''
    initialize centroids based on the number (k) and method specified
    Args:
        X: m x d array where m = # datapoints, d = dim
        K: # of clusters/initialized centroids
        dist type: l1 (manhattan) or l2 (euclidean)
        init_method: string. Possible methods = 'random_uniform', 'random_data', 'random_partition', 'plusplus'.
            random_uniform: random centroids initialized from uniform distribution between min and max values for each axis.
            random_data: centroids are picked randomly from the datapoints.
            random_partition: each datapoint is assigned random cluster, the centroids of which are assigned as initial centroids.
            plusplus: first datapoint is selected to be the first centroid. Subsequent centroid is selected from the remaining datapoints with probability proportional to min squared distance to the already picked centroids.
            rng: random number generator from numpy.random.default_rng() function
    Returns:
        initialized centroids: k x d array
    '''
    if not rng:
        rng = np.random.default_rng()
        
    if init_method == 'plusplus':
        seeds = rng.integers(10000, size=K)
        
        rng = np.random.default_rng(seeds[0])
        C = [X[rng.integers(X.shape[0])]]
        
        for k in range(1, K):
            
            if dist_type == 'l2':
                D2 = np.array([min([np.inner(c-x, c-x) for c in C]) for x in X])
            if dist_type == 'l1':
                D2 = np.array([min([((c-x).abs().sum())**2 for c in C]) for x in X])
                
            probs = D2/D2.sum()
            cumprobs = probs.cumsum()
            
            rng = np.random.default_rng(seeds[k])
            r = rng.random()
            for j, p in enumerate(cumprobs):
                if r < p:
                    i = j
                    break
            C.append(X[i])
        return np.array(C)
    
    if init_method == 'random_uniform':
        C = rng.uniform(low=X.min(axis=0), high=X.max(axis=0), size=(K, X.shape[1]))
        return C
    
    if init_method == 'random_data':
        i = rng.integers(X.shape[0], size=K)
        C = X[i, :]
        return C
    
    if init_method == 'random_partition':
        C = []
        clusters = rng.integers(K, size=X.shape[0])
        for k in range(K):
            
            if dist_type == 'l2':
                c = X[clusters == k, :].mean(axis=0)
            if dist_type == 'l1':
                c = np.median(X[clusters == k, :], axis=0)
            
            C.append(c)
        return np.array(C)
    
    if init_method == 'inverted_plusplus':
        seeds = rng.integers(10000, size=K)
        
        rng = np.random.default_rng(seeds[0])
        C = [X[rng.integers(X.shape[0])]]
        
        for k in range(1, K):
            
            if dist_type == 'l2':
                D2 = np.array([min([np.inner(c-x, c-x) for c in C]) for x in X])
            if dist_type == 'l1':
                D2 = np.array([min([((c-x).abs().sum())**2 for c in C]) for x in X])
                
            probs = D2/D2.sum()
            cumprobs = probs.cumsum()
            inverted_cumprobs = 1-cumprobs
            
            rng = np.random.default_rng(seeds[k])
            r = rng.random()
            for j, p in enumerate(inverted_cumprobs):
                if r < p:
                    i = j
                    break
            C.append(X[i])
        return np.array(C)
    

def compute_loss(X, closest_centroids, dist_type='l2'):
    if dist_type == 'l2':
        loss = ((X-closest_centroids)**2).sum()
    if dist_type == 'l1':
        loss = (((X-closest_centroids).sum())**2).sum()
    return loss
        

class KMeans:
    
    def __init__(self, arr):
        assert len(arr.shape) == 2
        self.data = arr
        self.dim = arr.shape[1]
        self.m = arr.shape[0]
        
    def run_kmeans(self, k, dist_type='l2', init_method='plusplus', rng=None):
        
        tic = process_time()    # start of initialization of centroids
        
        # initialization of centroids
        initial_centroids = initialize_centroids(self.data, k, init_method=init_method, rng=rng)
        
        # Loop to minimize KMeans loss
        toc = process_time()    # end of initialization, start of optimization
        clusters = None
        
        iteration = 0
        while True:
            clusters = run_step_A(self.data, initial_centroids, dist_type=dist_type)
            centroids = run_step_B(self.data, k, clusters, initial_centroids, dist_type=dist_type)
            iteration += 1
            
            if np.all(initial_centroids == centroids):
                
                closest_centroids = centroids[clusters]
                
                # compute loss, defined as sum of squared distance from datapoints to their corresponding centroids
                loss = compute_loss(self.data, closest_centroids, dist_type=dist_type)
                
                toc2 = process_time()    # end of optimization
                return {'clusters': clusters,
                        'centroids': centroids,
                        'closest_centroids': closest_centroids,
                        'loss': loss,
                        'initialization_time': toc - tic,
                        'optimization_time': toc2 - toc,
                        'iterations': iteration}
            else:
                initial_centroids = centroids
                
    def run_repeated_kmeans(self, k, n_repeats, dist_type='l2', init_method='plusplus', rng=None, verbose=True):
            
        # initalize empty list to host all the KMeans results
        runs = []
        
        for i in range (n_repeats):
            if verbose:
                print(f"running {i+1}-th iteration")
            result = self.run_kmeans(k, dist_type=dist_type, init_method=init_method, rng=rng)
            runs.append(result)
        
        df_runs = pd.DataFrame(runs)
        return df_runs