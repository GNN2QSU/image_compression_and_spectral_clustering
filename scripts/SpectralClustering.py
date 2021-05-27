from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scripts.KMeans import *


class SpectralClust:
    def __init__(self, edges):
        '''
        initialize SpectralClust class
        Args:
            edges: m x 2 ndarray where m = number of connections
                elements of edges array correspond to index of node
                example:
                    [[0, 1],     # node 0 connected to node 1
                     [1, 3]]     # node 1 connected to node 3
        '''
        self.edges = edges
        self.n = edges.max()+1
        self.U = None
        
    def construct_graph_laplacian(self):
        '''
        compute graph Laplacian (L) matrix
        Args:
            n: dimension of square matrix
            method: method used to compute L. Options are 'unnormalized' where L = D-A or 'normalized' where L = D^(-1/2)AD^(-1/2)
        Returns:
            L: n x n graph Laplacian matrix
        '''
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        v = np.ones(self.edges.shape[0])

        A = sparse.coo_matrix((v, (i, j)), shape=(self.n, self.n))
        
        A = (A + np.transpose(A))/2
        A = sparse.csc_matrix.todense(A)    # convert to dense matrix
        D = np.diag(1/np.sqrt(np.sum(A, axis=1)).A1)
        L = D @ A @ D
            
        return np.array(L)
    
    def construct_similarity_matrix(self, plot_eigenvalues=True, figsize=None):
        '''
        plot eigenvalues
        Args:
            method: method used to compute graph Laplacian matrix. 'normalized' or 'unnormalized'
            figsize: a tuple corresponding figure size
        Return:
            U: n x n similarity matrix containing full eigenvectors with sorted columns (descending for unnormalized, ascending for normalized)
        '''
        L = self.construct_graph_laplacian()
        v, x= np.linalg.eig(L)
        v = v.real
        x = x.real
        
        idx_sorted = np.argsort(-v)    # order of eigenvalues (highest to highest)
        x_axis = range(len(v))
        y_axis = v[idx_sorted]
        
        if plot_eigenvalues:
            plt.figure(figsize=figsize)
            plt.scatter(x=x_axis, y=y_axis, s=3)
            plt.title('Plot of Eigenvalues')
            plt.ylabel('lambda')
        U = normalize(x, axis=1)
        self.U = U
    
    def run_clustering(self, 
                       k, 
                       n_repeats, 
                       dist_type='l2', 
                       init_method='plusplus', 
                       rng=None,
                       verbose=False):
        '''
        cluster k selected columns from similarity matrix.
        Args:
            k: int, # of clusters
            n_repeats: # of KMeans with different initialization performed
            dist_type: 'l1' (manhattan) or 'l2' (euclidean)
            init_method: centroids initialization method:
                random_uniform: random centroids initialized from uniform distribution between min and max values for each axis.
                random_data: centroids are picked randomly from the datapoints.
                random_partition: each datapoint is assigned random cluster, the centroids of which are assigned as initial centroids.
                plusplus: first datapoint is selected to be the first centroid. Subsequent centroid is selected from the remaining datapoints with probability proportional to min squared distance to the already picked centroids.
            rng: random number generator from numpy.random.default_rng() function
            verbose: boolean, True = print iteration number
        '''
        if self.U is None:
            print('similarity matrix not found')
            return None
        
        U = self.U[:, :k]
        kmeans = KMeans(U)
        results = kmeans.run_repeated_kmeans(k, n_repeats, dist_type=dist_type, init_method=init_method, rng=rng, verbose=verbose)
        best_result = results.sort_values(by='loss').iloc[0].to_dict()
        return {'best_result': best_result, 'results': results}
    
    def run_clustering_once(self,
                            k,
                            dist_type='l2',
                            init_method='plusplus',
                            rng=None):
        '''
        run one-time clustering with only one initialization
        '''
        if self.U is None:
            print('similarity matrix not found')
            return None
        
        U = self.U[:, :k]
        kmeans = KMeans(U)
        results = kmeans.run_kmeans(k, dist_type=dist_type, init_method=init_method, rng=rng)
        return results