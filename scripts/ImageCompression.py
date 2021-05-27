from PIL import Image
import matplotlib.pyplot as plt
from scripts.KMeans import *

def compress_image(img_path, 
                   n_colors, 
                   n_repeats,
                   dist_type='l2',
                   init_method='plusplus',
                   rng=None, 
                   plot=True, 
                   save=True,
                   save_loc=None,
                   verbose=True):
    
    image = Image.open(img_path)
    arr = np.array(image)

    h = arr.shape[0]
    w = arr.shape[1]
    d = arr.shape[2]
    arr = arr.reshape(h*w, d)
    
    kmeans = KMeans(arr)
    results = kmeans.run_repeated_kmeans(n_colors, n_repeats, dist_type=dist_type, init_method=init_method, rng=rng, verbose=verbose)
    
    best_result = results.sort_values(by='loss').iloc[0].to_dict()
    compressed = ((best_result['closest_centroids']).reshape((h, w, d))).astype(np.uint8)
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize = (12, 8))
        ax[0].imshow(arr.reshape((h, w, d)))
        ax[0].set_title('Original Image')
        ax[1].imshow(compressed)
        ax[1].set_title('Compressed Image')
        for ax in fig.axes:
            ax.axis('off')
        plt.tight_layout();
    
    if save:
        plt.imsave(save_loc, compressed)
        
    return {'best_result': best_result, 'results': results}
