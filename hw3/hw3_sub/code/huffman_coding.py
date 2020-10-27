import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn
from heapq import heappush, heappop, heapify
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: encoding map and frequency map for the current weight layer.
    """
    # Your code: Implement the Huffman coding here. Store the encoding map into 'encoding'
    # and frequency map into 'frequency'.
    flatten_weights = weight.reshape(-1,1)
    counts = [(flatten_weights==center).sum() for center in centers]
    heap = [[count, [[center, ""]]] for (count, center) in zip(counts,centers)]
    heapify(heap)

    while len(heap) >= 2:
        item_S = heappop(heap)
        item_L = heappop(heap)    
        for code_S in item_S[1]: # for smaller item to add 1 to each of center
            code_S[1] = '1' + code_S[1]
        for code_L in item_L[1]: # for larger item to add 0 to each of center
            code_L[1] = '0' + code_L[1]
        new_item = [item_S[0] + item_L[0]] +  [item_S[1] + item_L[1]]
        heappush(heap,new_item)
    encoded = heappop(heap)[1]

    
    encodings = {}
    frequency = {}
    for row in encoded:
        encodings[row[0]] = row[1]
        
    for (count, center) in  zip(counts,centers):
        frequency[center] = count
 
    return encodings, frequency


def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param encodings: encoding map of the current layer w.r.t. weight (centriod) values.
    :param frequency: frequency map of the current layer w.r.t. weight (centriod) values.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    total_bits=0
    total_param=0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
            total_bits=total_bits+weight.size*huffman_avg_bits
            total_param=total_param+weight.size
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
            total_bits=total_bits+weight.size*huffman_avg_bits
            total_param=total_param+weight.size
    print('the average bit length for the weight parameters: ',total_bits/total_param)

    return freq_map, encodings_map