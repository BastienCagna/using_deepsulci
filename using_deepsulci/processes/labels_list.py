'''
    Labels list
'''

from __future__ import print_function
from __future__ import absolute_import
import traits.api as traits
import numpy as np
import sigraph
import os
import pandas as pd
from capsul.api import Process
from soma import aims
from deepsulci.deeptools.dataset import extract_data
from joblib import Parallel, delayed, cpu_count


def extract_names(gfile, i, total, flt=None):
    print("Reading file {} / {}".format(i, total))
    graph = aims.read(gfile)
    if flt is not None:
        flt.translate(graph)

    data = extract_data(graph)
    return np.asarray(data['names'])


class SulciList(Process):
    '''
    Process to list given and missing labels

    **Warning:** Graphs should be of the same side!

    '''

    def __init__(self):
        super(SulciList, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False),
                                             desc='training base graphs'))
        self.add_trait('graphs_notcut', traits.List(
            traits.File(output=False),
            desc='training base graphs before manual cutting of the'
                 ' elementary folds'))
        self.add_trait('translation_file', traits.File(
            output=False, optional=True,
            desc='file (.trl) containing the translation of the sulci to'
                 'applied on the training base graphs (optional)'))

        self.add_trait('report', traits.File(
            output=True, desc='file (.csv) stats'))

    def _run_process(self):
        agraphs = np.asarray(self.graphs)
        agraphs_notcut = np.asarray(self.graphs_notcut)

        if os.path.exists(self.translation_file):
            flt = sigraph.FoldLabelsTranslator()
            flt.readLabels(self.translation_file)
            trfile = self.translation_file
        else:
            flt, trfile = None, None
            print('Translation file not found.')

        # Read all graphs
        n = max(cpu_count() - 2, 1)
        names_arrays = Parallel(n_jobs=n)(delayed(extract_names) \
              (gfile, ig+1, len(agraphs), flt) for ig, gfile in enumerate(agraphs))

        sulci_side_list = set()
        for l in names_arrays:
            for name in l:
                sulci_side_list.add(name)

        counts = {k: [] for k in ['graph'] + list(sulci_side_list)}
        for ig, fname in enumerate(agraphs):
            counts['graph'].append(fname)
            for ss in sulci_side_list:
                counts[ss].append(np.sum(names_arrays[ig] == ss))
        df = pd.DataFrame(counts)
        df.to_csv(self.report)
