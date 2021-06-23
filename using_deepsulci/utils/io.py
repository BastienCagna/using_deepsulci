import numpy as np
import os.path as op
import sigraph
from soma import aims
from deepsulci.deeptools.dataset import extract_data


def load_graphs(graphs, translation_file=None, verbosity=0):
    # Load graphs
    agraphs = np.asarray(graphs)

    if op.exists(translation_file):
        flt = sigraph.FoldLabelsTranslator()
        flt.readLabels(translation_file)
        trfile = translation_file
    else:
        flt, trfile = None, None
        print('Translation file not found.')

    # Read all graphs
    sulci_side_list = set()
    dict_bck2 = {}
    dict_names = {}
    for gfile in agraphs:
        if verbosity > 0:
            print("Reading", gfile)
        graph = aims.read(gfile)
        if trfile is not None:
            flt.translate(graph)

        data = extract_data(graph)
        dict_bck2[gfile] = data['bck2']
        dict_names[gfile] = data['names']
        for n in data['names']:
            sulci_side_list.add(n)

    return dict_bck2, dict_names
