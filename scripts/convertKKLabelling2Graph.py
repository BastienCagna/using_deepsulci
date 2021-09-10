import os.path as op
from collections import Counter
from os import listdir, makedirs
from soma import aims
import numpy as np
import pandas as pd
import nibabel.gifti as ng
from sklearn.neighbors import KNeighborsClassifier
import json
from scipy.spatial.distance import euclidean


def find_key(input_dict, value):
    return next((k for k, v in input_dict.items() if v == value), None)


def list_subjects(path):
    subjects = []
    for f in listdir(path):
        if op.isdir(op.join(path, f)):
            subjects.append(f)
    return list(sorted(subjects))


def read_labels_keys(labels_f):
    labels_df = pd.read_csv(labels_f, sep="\t", header=None)
    return {k: l for (l, k) in zip(labels_df[0], labels_df[1])}


def read_graph(graph_f, sulci_pointset_key='aims_other'):
    graph = aims.read(graph_f)
    vs = graph['voxel_size']

    # For each sulci of the graph
    sulci_pointsets = []
    for vert in graph.vertices():
        coords = []
        for k in ['aims_bottom', 'aims_other', 'aims_ss']:
            if k in vert.keys():
                for point in vert[k][0].keys():
                    coords.append(list(p * v for p, v in zip(point, vs)))
        sulci_pointsets.append(coords)
    return graph, sulci_pointsets


def convert_graph(gii_mesh_f, gii_texture_f, insula_texture_f, cing_texture_f,
                  graph_f, labels_f, new_graph_f, insula_key=100,
                  cingular_key=101):
    mesh = aims.read(gii_mesh_f)

    # Load and merge all labelling textues
    labels_tex = ng.read(gii_texture_f).darrays[0].data#np.array(aims.read(gii_texture_f)[0])
    labels_tex = np.round(labels_tex).astype(int) # np.array(labels_tex, dtype=int)
    #insula_tex = np.array(aims.read(insula_texture_f)[0])
    #cingular_tex = np.array(aims.read(cing_texture_f)[0])
    #labels_tex[insula_tex == 100] = insula_key
    #labels_tex[cingular_tex == 100] = cingular_key

    # Load all the labels/values correspondances
    keys_to_labels = read_labels_keys(labels_f)
    hemi_sfx = list(keys_to_labels.values())[0].split('.')[1]
    keys_to_labels[insula_key] = 'insula.' + hemi_sfx
    keys_to_labels[cingular_key] = 'cingular.' + hemi_sfx

    # Read the graph and extract list of vertex coordinates for each sulci
    graph, sulci_pointsets = read_graph(graph_f)

    # Fit a KNeighbors classifier
    sel = labels_tex != -2#find_key(keys_to_labels, "unknown")
    labels = np.unique(labels_tex[sel])
    clf = KNeighborsClassifier(n_neighbors=len(labels), weights='distance', metric="chebyshev")
    clf.fit(np.array(mesh.vertex())[sel], labels_tex[sel])

    # Then estimate the sulci name by finding the nearest drawn sulcal line
    scores = {}
    for iv, vert in enumerate(graph.vertices()):
        if sulci_pointsets[iv]:
            # Probability of each label for each vertex of the sulci
            preds_p = clf.predict_proba(sulci_pointsets[iv])
            kng_dist, _ = clf.kneighbors(sulci_pointsets[iv])
            # Prediction for each vertex (maximal probability)
            if np.mean(preds_p[:, 0]) < 1:
                preds_p[:, 0] = 0
            preds = np.argmax(preds_p / kng_dist, axis=1)
            # Group prediction is the one that appear the most
            votes = Counter(preds)
            keys = list(votes.keys())
            if len(keys) > 1 and 0 in keys:
                votes.pop(0)
            pred = max(votes, key=votes.get)
            # Compute average probability of vertex that are close to the pred
            # avg_p = np.mean(preds_p[preds == pred, pred])
            # votes_values = np.array(list(v for v in votes.values()))
            # avg_p = votes_values.max() / votes_values.sum()
            n = preds_p.shape[0]
            scr_p = np.sum(preds_p[:, pred] > .5) / n
            #kng_dist.sort(axis=0)
            scr_d = (np.sum(kng_dist[:, pred] < 12) - np.sum(kng_dist[:, pred] < 7))/ n#1 / np.mean(kng_dist[:, pred])
            scr = scr_p#max(scr_d, scr_p)
            vert['name'] = keys_to_labels[labels[pred]] #if avg_p > .2 else "unknown"

            #print(vert['name'], scr_d, scr_p)
            scores[vert['index']] = {'scores': [scr]}
    # Save the graph and the scores
    aims.write(graph, new_graph_f)
    with open(new_graph_f[:-3] + 'json', 'w') as f:
        json.dump({"meta": { "labels": ['Label probability']}, "vdata": scores}, f)


def main():
    db_dir = "/neurospin/dico/data/bv_databases/baboon/marsBaboons"
    txt_pattern = db_dir + "/Adrien/[sub]/t1mri/default_acquisition/default_analysis/segmentation/mesh/surface_analysis/[sub]_[hemi]graphLabelBasins.txt"
    gii_pattern = db_dir + "/Adrien/[sub]/t1mri/default_acquisition/default_analysis/segmentation/mesh/surface_analysis/[sub]_[hemi]white_sulcalines_editedbyKK.gii"
    giii_pattern = db_dir + "/Adrien/[sub]/t1mri/default_acquisition/default_analysis/segmentation/mesh/surface_analysis/[sub]_[hemi]white_pole_insula_KK.gii"
    giic_pattern = db_dir + "/Adrien/[sub]/t1mri/default_acquisition/default_analysis/segmentation/mesh/surface_analysis/[sub]_[hemi]white_pole_cingular_KK.gii"
    mesh_pattern = db_dir + "/Adrien/[sub]/t1mri/default_acquisition/default_analysis/segmentation/mesh/[sub]_[hemi]white_fine.gii"
    arg_pattern = db_dir + "/Adrien/[sub]/t1mri/default_acquisition/default_analysis/folds/3.1/[hemi][sub].arg"
    #out_pattern = "/neurospin/dico/bcagna/data/graph_conversion/marsBaboons/[sub]/[hemi][sub].arg"
    out_pattern = db_dir + "/Adrien/[sub]/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_KK/[hemi][sub]_default_session_KK.arg"
    outm_pattern = "/neurospin/dico/bcagna/data/graph_conversion/marsBaboons/[sub]/[sub][hemi]white_fine.gii"

    subs = list_subjects(op.join(db_dir, "Adrien"))
    subs = ['_session_01_subject_hunt']
    for i_s, sub in enumerate(subs[-1:]):
        for h in ['L', 'R']:
            print("subject {}/{} {} ({})".format(i_s+1, len(subs), h, sub))
            out_f = out_pattern.replace("[sub]", sub).replace("[hemi]", h)
            mesh_f = mesh_pattern.replace("[sub]", sub).replace("[hemi]", h)
            makedirs(op.split(out_f)[0], exist_ok=True)
            convert_graph(
                mesh_f,
                gii_pattern.replace("[sub]", sub).replace("[hemi]", h),
                giii_pattern.replace("[sub]", sub).replace("[hemi]", h),
                giic_pattern.replace("[sub]", sub).replace("[hemi]", h),
                arg_pattern.replace("[sub]", sub).replace("[hemi]", h),
                txt_pattern.replace("[sub]", sub).replace("[hemi]", h),
                out_f,
            )

            # mesh = ng.read(mesh_f)
            # mesh = ng.GiftiImage(darrays=[
            #     ng.GiftiDataArray(mesh.darrays[0].data,
            #                       intent='NIFTI_INTENT_POINTSET',
            #                       encoding="GIFTI_ENCODING_B64BIN"),
            #     ng.GiftiDataArray(mesh.darrays[1].data,
            #                       intent='NIFTI_INTENT_TRIANGLE',
            #                       encoding="GIFTI_ENCODING_B64BIN")
            # ])
            # ng.write(mesh, outm_pattern.replace("[sub]", sub).replace("[hemi]", h))


if __name__ == "__main__":
    main()
