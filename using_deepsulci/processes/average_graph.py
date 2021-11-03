from collections import Counter
import numpy as np
from tqdm import tqdm
from soma import aims
from sklearn.neighbors import NearestNeighbors
import nibabel.gifti as ng

from os import listdir
import os.path as op

import anatomist.api as ana
from soma.qt_gui.qt_backend import Qt
import sys
import colorado as cld
from joblib.parallel import Parallel, delayed, cpu_count
from time import time


def load_labels_coords(g, label_key='name', bucket_key='aims_ss'):
    graph = aims.read(g)
    trans_tal = aims.GraphManip.talairach(graph)

    pointsets = {}
    for v in graph.vertices():
        if label_key in v.keys() and bucket_key in v.keys():
            points = []
            for p in v[bucket_key][0].keys():
                p0 = [p * v for p, v in zip(p, graph['voxel_size'])]
                p1 = trans_tal.transform(p0)
                points.append((p1[0], p1[1], p1[2]))

            label = v[label_key]
            if label in pointsets.keys():
                pointsets[label].extend(points)
            else:
                pointsets[label] = points
    return pointsets


def concat_meshes(gii):
    pointset = []
    triangles = []
    len_mem = 0
    n_pointsets = 0
    n_triangles = 0
    index = []
    for arr in gii.darrays:
        if arr.intent == 1008:
            len_mem = len(pointset)
            pointset.extend(arr.data)
            index.extend([n_pointsets] * len(arr.data))
            n_pointsets += 1
        elif arr.intent == 1009:
            offset = len_mem if n_pointsets > n_triangles else len(pointset)
            triangles.extend(arr.data + offset)
            n_triangles += 1
    if len(pointset) == 0:
        return None
    mesh = ng.GiftiImage(darrays=[
        ng.GiftiDataArray(pointset, intent='NIFTI_INTENT_POINTSET'),
        ng.GiftiDataArray(triangles, intent='NIFTI_INTENT_TRIANGLE'),
    ], header=gii.get_header(), meta=gii.get_meta())

    texture = ng.GiftiImage(darrays=[ng.GiftiDataArray(
        np.array(index, dtype=float), intent='NIFTI_INTENT_LABEL')],
        header=gii.get_header(), meta=gii.get_meta())
    return mesh, texture


def average(graphs_f):
    """
        Align all the graph
    """
    # Load graph
    print("Extracting coordinates from each graphs and labels...")
    data = Parallel(n_jobs=max(2, 1))(
        delayed(load_labels_coords)(g) for g in graphs_f)
    # data = list(load_labels_coords(g) for g in graphs_f)

    # List used labels
    arr = []
    for g in data:
        arr.extend(list(g.keys()))
    labels = list(sorted(set(arr)))
    print("Used labels:", labels)

    print("Grouping pointsets...")
    pointsets = {label: [] for label in labels}
    for pointset in data:
        for label in pointset.keys():
            pointsets[label].extend(pointset[label])

    # Create a mesh for each label
    darrays = []
    for il, label in enumerate(labels[:6]):
        print("Meshing label:", label)
        pointset = np.array(list(set(pointsets[label])))
        if len(pointset) > 0:
            mesh = cld.aims_tools.bucket_to_mesh(
                pointset, smoothingFactor=2, aimsThreshold="95%",
                deciMaxError=2, deciMaxClearance=6, smoothIt=200)
            print("{} points -> {} vertices".format(
                len(pointset), len(list(mesh.vertex()))))
            darrays.extend([
                ng.GiftiDataArray(list(mesh.vertex()), 'NIFTI_INTENT_POINTSET'),
                ng.GiftiDataArray(list(mesh.polygon()), 'NIFTI_INTENT_TRIANGLE')
            ])

    mesh_gii = ng.GiftiImage(darrays=darrays)
    cmesh_gii, tex_gii = concat_meshes(mesh_gii)

    tex_f = '/var/tmp/average_folding_texture.gii'
    ng.write(tex_gii, tex_f)
    cmesh_f = '/var/tmp/average_folding_concat_mesh.gii'
    ng.write(cmesh_gii, cmesh_f)
    mesh_f = '/var/tmp/average_folding_mesh.gii'
    ng.write(mesh_gii, mesh_f)

    show_fusion(cmesh_f, tex_f)


def show_fusion(mesh_f, texture_f):
    a = ana.Anatomist()
    mesh = a.loadObject(mesh_f)
    tex = a.loadObject(texture_f)
    fusion = a.fusionObjects([mesh, tex], 'FusionTexSurfMethod')

    win = a.createWindow('3D')
    a.addObjects(fusion, win)

    Qt.QApplication.instance().exec_()


def main():
    d = "/neurospin/dico/bcagna/data/deepsulci_basic_learning/evaluations/" \
        "cohort-Archi_hemi-L_model-unet3d_d00b01/run-01"
    graphs = []
    for f in listdir(d):
        if f.endswith(".arg"):
            graphs.append(op.join(d, f))
    graphs = list(sorted(graphs))

    average(graphs[:2])


if __name__ == "__main__":
    main()
