from collections import Counter
import numpy as np
from tqdm import tqdm
from soma import aims
from sklearn.neighbors import NearestNeighbors
import nibabel.gifti as ng

from os import listdir
import os.path as op


# def merge_labels(graph):
#     labels = []
#     for v in g.vertices():
#         if 'name' in v.keys():
#             labels.append(v['name'])
#     labels = Counter(labels)
#
#     for label in labels.keys():
#         if labels[label] > 1:
#             bck = None
#             for v in g.vertices():
#                 if 'name' in v.keys() and v['name'] == label:
#                     if bck is None:
#                         bck = v
#                     else:
#                         pass

def compute_triangles(pointset):
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(pointset)
    _, indices = nbrs.kneighbors(pointset)
    return indices


def average(graphs_f):
    # Load graph
    graphs = list(aims.read(g) for g in graphs_f)

    # # Merge labels
    # for g in graphs:
    #     g = merge_labels(g)

    # Load labels
    labels = []
    for g in graphs:
        for v in g.vertices():
            if 'name' in v.keys():
                labels.append(v['name'])

    darrays = []
    for label in tqdm(list(set(labels))[:4]):
        points = []
        for g in graphs:
            trans_tal = aims.GraphManip.talairach(g)
            for v in g.vertices():
                if 'name' in v.keys() and v['name'] == label and \
                        'aims_ss' in v.keys():
                    for p in v['aims_ss'][0].keys():
                        p0 = [p * v for p, v in zip(p, g['voxel_size'])]
                        p1 = trans_tal.transform(p0)
                        points.append((p1[0], p1[1], p1[2]))

        pointset = np.array(list(set(points)))
        if len(pointset) > 0:
            triangles = compute_triangles(pointset)
            darrays.append(ng.GiftiDataArray(pointset, 'NIFTI_INTENT_POINTSET'))
            darrays.append(ng.GiftiDataArray(triangles, 'NIFTI_INTENT_TRIANGLE'))
    gii = ng.GiftiImage(darrays=darrays)
    ng.write(gii, '/var/tmp/average_folding_mesh.gii')



def main():
    d = "/neurospin/dico/bcagna/data/deepsulci_basic_learning/evaluations/" \
        "cohort-Archi_hemi-L_model-unet3d_d00b01/run-01"
    graphs = []
    for f in listdir(d):
        if f.endswith(".arg"):
            graphs.append(op.join(d, f))

    average(graphs[:2])

if __name__ == "__main__":
    main()
