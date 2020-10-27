import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def index_of_first(lst, prod):
    for i,v in enumerate(lst):
        if v >= prod:
            return i

def _cdf(x, data):
    return (sum(data<=x)/data.size)

def _quantize_layer(weight, bits=8):
    """
    :param weight: A numpy array of any shape.
    :param bits: quantization bits for weight sharing.
    :return quantized weights and centriods.
    """
    # Your code: Implement the quantization (weight sharing) here. Store 
    # the quantized weights into 'new_weight' and store kmeans centers into 'centers_'
    
    
    
    
    flatten_weights = weight.reshape(-1,1)
    index_nonzero = np.nonzero(flatten_weights!=0)[0] 
    weights_nonzero = flatten_weights[index_nonzero] # Avoid the cluster with label 0 
    
    #Density-Based init  
#     prods = np.linspace(0, 100, num = 2**bits+2)
#     prods = prods[1:-1]
#     init = np.array([np.percentile(weights_nonzero, prod) for prod in prods]).reshape(-1, 1)
    #Random
#     init = 'random'
    
    #Linear
    init = np.linspace(min(weights_nonzero), max(weights_nonzero), num = 2**bits) #linear 

    kmeans = KMeans(n_clusters=2**bits, init = init, n_init = 1, random_state=0).fit(weights_nonzero)
    new_weight = flatten_weights.copy()
    centers_ = kmeans.cluster_centers_
    
    if 0 in centers_: # Avoid the cluster with label 0 
        centers_[centers_==0] = [1e-8]*((centers_== 0).sum())
    
    new_weight[index_nonzero] = np.array([centers_[i][0] for i in kmeans.labels_]).reshape(len(index_nonzero),-1)
    new_weight = new_weight.reshape(weight.shape)

    return new_weight, centers_

def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.conv.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.linear.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

