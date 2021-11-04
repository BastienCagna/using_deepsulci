from os import listdir, makedirs
import os.path as op
import numpy as np
import json
from soma import aims


class SubjectDataset:
    def __init__(self, name, t1, roots, skeleton, graph, notcut_graph):
        # grey_white, hemi_cortex, split_brain, white_mesh, pial_mesh):
        self.name = name
        self.t1 = t1
        self.roots = roots
        self.skeleton = skeleton
        self.graph = graph
        self.notcut_graph = notcut_graph
        # self.grey_white = grey_white
        # self.hemi_cortex = hemi_cortex
        # self.split_brain = split_brain
        # self.white_mesh = white_mesh
        # self.pial_mesh = pial_mesh

    def __lt__(self, other):
        return self.name < other.name

    def check(self):
        if self.notcut_graph and not op.exists(self.notcut_graph):
            raise IOError("Missing file: " + self.notcut_graph)
        for f in [self.t1, self.roots, self.skeleton, self.graph,
                  # self.grey_white, self.split_brain, self.white_mesh,
                  # self.pial_mesh
                  ]:
            if not op.exists(f):
                raise IOError("Missing file: " + f)
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


class Cohort(object):
    def __init__(self, name="Unnamed", hemi="X", subjects=[], from_json=None, check=True):
        if name is None and subjects is None and from_json is None:
            raise ValueError("Cannot create Cohort without inputs.")
        elif from_json is None:
            self.name = name
            self.hemi = hemi
            self.subjects = subjects

            if check:
                for s in subjects:
                    s.check()
        else:
            with open(from_json, 'r') as infile:
                data = json.load(infile)
                self.name = data["name"]
                self.hemi = data["hemi"]
                self.subjects = []
                for s in data["subjects"]:
                    sub = SubjectDataset(
                        s['name'], s['t1'], s['roots'], s['skeleton'],
                        s['graph'], s['notcut_graph'],
                        # s['grey_white'], s['hemi_cortex'], s['split_brain'],
                        # s['white_mesh'], s['pial_mesh']
                    )
                    if check:
                        sub.check()
                    self.subjects.append(sub)

    def __iter__(self):
        return CohortIterator(self)

    def __len__(self):
        return len(self.subjects)

    def __contains__(self, subject):
        if isinstance(subject, str):
            for s in self.subjects:
                if s.name == subject:
                    return True
            return False
        return subject in self.subjects

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
                "notcut_graph": s.notcut_graph,
                # "grey_white": s.grey_white,
                # "hemi_cortex": s.hemi_cortex,
                # "split_brain": s.split_brain,
                # "white_mesh": s.white_mesh,
                # "pial_mesh": s.pial_mesh
            })
        data = {"name": self.name, "hemi": self.hemi, "subjects": subdata}

        if filename:
            dir_path, _ = op.split(filename)
            makedirs(dir_path, exist_ok=True)
            with open(filename, 'w') as outfile:
                json.dump(data, outfile)
        return data

    # def get_sulci_side_list(self):
    #     for s in self.subjects:


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
        seg_dir = op.join(db_dir, scenters[i], s, 't1mri', acquisition,
                          analysis, 'segmentation')
        roots = op.join(seg_dir, hemi + 'roots_' + s + '.nii.gz')

        # Skeleton
        skeleton = op.join(seg_dir, hemi + 'skeleton_' + s + '.nii.gz')

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

        # Grey white
        # gw = op.join(seg_dir, hemi + 'grey_white_' + s + '.nii.gz')
        # gw = op.join(seg_dir, hemi + 'grey_white_' + s + '.nii.gz')

        subjects.append(SubjectDataset(s, t1, roots, skeleton, gfile, ngfile,
                                       # gw, hs, sb, white_m, pial_m
                                       ))
    return Cohort(name, hemi, subjects)


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
