import os
from os.path import exists
import numpy as np
import sys

def output_file(a, idx2name, c_idx, node_file, edge_file):

    with open(edge_file, 'w') as fid:
        fid.write('Source\tTarget\n')
        for i in range(len(a)):
            fid.write(f'{a[i,0]}\t{a[i,1]}\n')

    with open(node_file, 'w') as fid:
        fid.write('Id\tLabel\tColor\n')
        for i in range(len(idx2name)):
            fid.write(f'{i}\t{idx2name[i]}\t{c_idx[i]}\n')
            

def read_team_name(f_path):
    # read inverse_teams.txt file
    idx2name = []
    with open(f_path) as fid:
        for line in fid.readlines():
            name = line.split("\t", 1)[1]
            idx2name.append(name[:-1])
    return idx2name


def import_graph(f_path):
    # read the graph from 'play_graph.txt'
    with open(f_path) as graph_file:
        lines = [line.split() for line in graph_file]
    return np.array(lines).astype(int)