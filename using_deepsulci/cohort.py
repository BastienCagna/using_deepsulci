from os import listdir
import os.path as op
import numpy as np
import json
from soma import aims


class SubjectDataset:
    def __init__(self, name, t1, roots, skeleton, graph, notcut_graph,
                 replace_roots=True):
        self.name = name
        self.t1 = t1
        if op.exists(roots):
            self.roots = roots
        elif replace_roots:
            print("/!\\ [subject " + name +
                  "] roots image doesn't exist, replaced by skeleton image")
            self.roots = skeleton
        self.skeleton = skeleton
        self.graph = graph
        self.notcut_graph = notcut_graph

    def __lt__(self, other):
        return self.name < other.name

    def check(self):
        if not op.exists(self.t1):
            raise IOError("Missing file: " + self.t1)
        if not op.exists(self.roots):
            raise IOError("Missing file: " + self.roots)
        if not op.exists(self.skeleton):
            raise IOError("Missing file: " + self.skeleton)
        if not op.exists(self.graph):
            raise IOError("Missing file: " + self.graph)
        if self.notcut_graph and not op.exists(self.notcut_graph):
            raise IOError("Missing file: " + self.notcut_graph)
        if not isinstance(self.name, str):
            raise ValueError("Name must be a string")


class CohortIterator:
    def __init__(self, cohort):
        self._cohort = cohort
        self._index = 0

    def __next__(self):
        item = self._cohort.subjects[self._index]
        self._index += 1
        return item


class Cohort:
    def __init__(self, name="Unnamed", subjects=[], from_json=None, check=True):
        if name is None and subjects is None and from_json is None:
            raise ValueError("Cannot create Cohort without inputs.")
        elif from_json is None:
            self.name = name
            self.subjects = subjects

            if check:
                for s in subjects:
                        s.check()
        else:
            with open(from_json, 'r') as infile:
                data = json.load(infile)
                self.name = data["name"]
                self.subjects = []
                for s in data["subjects"]:
                    sub = SubjectDataset(
                        s['name'], s['t1'], s['roots'], s['skeleton'],
                        s['graph'], s['notcut_graph'])
                    if check:
                        sub.check()
                    self.subjects.append(sub)

    def __iter__(self):
        return CohortIterator(self)

    def __len__(self):
        return len(self.subjects)

    def get_by_name(self, name):
        for s in self.subjects:
            if s.name == name:
                return s

    def get_graphs(self):
        graphs = []
        for s in self.subjects:
            graphs.append(s.graph)
        return graphs

    def get_notcut_graphs(self):
        graphs = []
        for s in self.subjects:
            if not s.notcut_graph:
                return []
            graphs.append(s.notcut_graph)
        return graphs

    def concatenate(self, cohort, new_name=None):
        new_name = self.name if new_name is None else new_name
        return Cohort(new_name, sorted(self.subjects + cohort.subjects))

    def to_json(self, filename=None):
        subdata = []
        for s in self.subjects:
            subdata.append({
                "name": s.name,
                "t1": s.t1,
                "roots": s.roots,
                "skeleton": s.skeleton,
                "graph": s.graph,
                "notcut_graph": s.notcut_graph
            })
        data = {"name": self.name, "subjects": subdata}
        if filename:
            with open(filename, 'w') as outfile:
                json.dump(data, outfile)
        return data


def bv_cohort(name, db_dir, hemi, centers, acquisition="default_acquisition",
              analysis="default_analysis", graph_v="3.3", ngraph_v="3.2",
              session="default_session", inclusion=[], exclusion=[]):
    """
    Parameters:
        db_dir: Brainvisa database directory
        hemi: Hemisphere ("L" or "R")
        centers: str or array
        acquisition:
        analysis:
        graph_v: Graph  version
        ngraph_v: Notcut graph version (same as graph if None, if -1, do not use
                  not cut graph)
        session: Labelling session
    """
    centers = [centers] if isinstance(centers, str) else centers
    ngraph_v = graph_v if ngraph_v is None else ngraph_v

    # List subjects
    snames = []
    scenters = []
    for center in centers:
        for f in listdir(op.join(db_dir, center)):
            if op.isdir(op.join(db_dir, center, f)) and \
                    (len(inclusion) == 0 or f in inclusion) and \
                    f not in exclusion:
                snames.append(f)
                scenters.append(center)
    order = np.argsort(snames)
    snames = np.array(snames)[order]
    scenters = np.array(scenters)[order]

    subjects = []
    for i, s in enumerate(snames):
        # T1
        t1 = op.join(
            db_dir, scenters[i], s, 't1mri', acquisition, s + ".nii.gz"
        )

        # Roots
        roots = op.join(
            db_dir, scenters[i], s, 't1mri', acquisition, analysis,
            'segmentation', hemi + 'roots_' + s + '.nii.gz'
        )

        # Skeleton
        skeleton = op.join(
            db_dir, scenters[i], s, 't1mri', acquisition, analysis,
            'segmentation', hemi + 'skeleton_' + s + '.nii.gz'
        )

        # Graph
        gfile = op.join(
            db_dir, scenters[i], s, 't1mri', acquisition, analysis, 'folds',
            graph_v, session, hemi + s + '_' + session + '.arg'
        )

        # Not cut graph
        if ngraph_v == -1:
            ngfile = None
        else:
            ngfile = op.join(
                db_dir, scenters[i], s, 't1mri', acquisition, analysis, 'folds',
                ngraph_v, hemi + s + '.arg'
            )

        subjects.append(SubjectDataset(s, t1, roots, skeleton, gfile, ngfile))
    return Cohort(name + "_hemi-" + hemi, subjects)


# TODO: remove following lines
# def archi_cohort(data_dir, hemi):
#     # List subjects for archi database
#     snames = []
#     for f in listdir(op.join(data_dir, "t1-1mm-1")):
#         if len(f) == 3:
#             snames.append(f)
#     snames = sorted(snames)
#
#     subjects = []
#     for s in snames:
#         gfile = op.join(
#             data_dir, 't1-1mm-1', s, 't1mri' + 'default_acquisition',
#             'default_analysis', 'folds', '3.3', 'session1_manual',
#             hemi + s + '_session1_manual.arg')
#         subjects.append(SubjectDataset(s, gfile))
#     return Cohort("Archi_hemi-" + hemi, subjects)
#
#
# def pclean_cohort(data_dir, hemi):
#     # List subjects for archi database
#     pclean = []
#     pclean_dirs = []
#     for d in ['jumeaux', 'nmr', 'panabase']:
#         for f in listdir(op.join(data_dir, "data", "database_learnclean", d)):
#             if op.isdir(f):
#                 pclean.append(f)
#                 pclean_dirs.append(d)
#     order = np.argsort(pclean)
#     pclean = np.array(pclean)[order]
#     pclean_dirs = np.array(pclean_dirs)[order]
#
#     subjects = []
#     for i, s in enumerate(pclean):
#         gfile = op.join(
#             data_dir, pclean_dirs[i], s , 't1mri', 't1', 'default_analysis',
#             'folds', '3.3', 'base2018_manual', hemi + s + '_base2018_manual.arg'
#         )
#         subjects.append(SubjectDataset(s, gfile))
#     return Cohort("PClean_hemi-" + hemi, subjects)
#
#
# def hcp_cohort(data_dir, hemi):
#     # List subjects for archi database
#     snames = []
#     for f in listdir(op.join(data_dir, "t1-1mm-1")):
#         if len(f) == 3:
#             snames.append(f)
#     snames = sorted(snames)
#
#     subjects = []
#     for s in snames:
#         gfile = op.join(
#             data_dir, 't1-1mm-1', s , 't1mri', 'default_acquisition',
#             'default_analysis', 'folds', '3.1', 'default_session_auto',
#             hemi + s + '_default_session_auto.arg'
#         )
#         subjects.append(SubjectDataset(s, gfile))
#     return Cohort("HCP_hemi-" + hemi, subjects)
