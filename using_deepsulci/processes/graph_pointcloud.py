#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
from soma import aims, aimsalgo
import pandas
import numpy as np
import subprocess
import six
from six.moves import range
from six.moves import zip


def get_vertex_points(v):
    ''' get array of voxels coordinates in buckets of the given graph node.
    (looks in ss, bottom, other buckets)
    '''
    ss = v.get('aims_ss')
    bottom = v.get('aims_bottom')
    other = v.get('aims_other')
    ss_pts = np.zeros((4, 0), dtype=int)
    bottom_pts = np.zeros((4, 0), dtype=int)
    other_pts = np.zeros((4, 0), dtype=int)
    if ss is not None and len(ss[0]) != 0:
        ss_pts = np.array(list(ss[0].keys()))
        ss_pts = np.hstack((ss_pts,
                            np.zeros((ss_pts.shape[0], 1), dtype=int))).T
    if bottom is not None and len(bottom[0]) != 0:
        bottom_pts = np.array(list(bottom[0].keys()))
        bottom_pts = np.hstack((bottom_pts,
                                np.zeros((bottom_pts.shape[0], 1),
                                         dtype=int))).T
    if other is not None and len(other[0]) != 0:
        other_pts = np.array(list(other[0].keys()))
        other_pts = np.hstack((other_pts,
                               np.zeros((other_pts.shape[0], 1), dtype=int))).T
    apts = np.hstack((ss_pts, bottom_pts, other_pts))
    return apts


def maj_label(pts, avol, get_counts=False):
    ''' Majority label in region pts of label volume avol.
    '''
    labels = np.unique(avol[pts])
    counts = [len(np.where(avol[pts] == l)[0]) for l in labels]
    imax = np.argmax(counts)
    if get_counts:
        return (labels[imax], labels, counts)
    return labels[imax]


def get_junction_points(apts, avor, askel, vs):
    # TODO: check if 6-connectivity is OK
    convmask = ([-1, 0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 1, 0, 0],
                [0, 0, -1, 0], [0, 0, 1, 0])  # 6-connectivity
    split_bk = aims.BucketMap_VOID()
    split_bk.header()['voxel_size'] = vs
    sb0 = split_bk[0]
    for p in apts.T:
        # Get roots values around the point
        p2 = np.vstack([p + m for m in convmask]).T
        vals = np.unique(avor[tuple(p2)])
        # Remove background value
        # TODO: check if there is actually -1 value (root background look to be -46)
        if -1 in vals:
            vals = [val for val in vals if val >= 0]
        # If the neightboors contain others values, this is a junction
        if len(vals) > 1:
            if askel is not None:
                askel[tuple(p)] = JUNCTION
            sb0[p[:3]] = 1
    return split_bk


def build_split_graph(graph, data, roots, skel=None, verbose=False):
    ''' Split graph vertices according to sub-vertex label classification
    (from Léonie Borne's automatic labelings).

    This goes throught the graph and check is several values are attributed
    to each sulcus. If it does, it list junctions voxels of the sulcus and
    split it in two parts and iterate if it is necessary (sub sulcus have
    more than loc_mis_max voxel that are not equal to majority label).

    Parameters
    ----------
    graph: aims.Graph
        cortical folds graph
    data: pandas.DataFrame
        voxel-wise data. Row should include "point_x", "point_y", "point_z"
        for voxels coordinates *in Talairach space* (they are transformed back
        into native space during the process), and "after_cutting" for labels.
    roots: aims.Volume_S16
        folds skeleton roots regions, from Morphologist pipeline.
    skel: aims.Volume_S16 (optional)
        folds skeleton, from Morphologist pipeline. If provided, the skeleton
        will be modified with added junction points.
    verbose: boolean (optional)
        Print detailed infos.

    Returns
    -------
    graph: aims.Graph
        the input graph, modified. Note that the input graph is not copied
        but modified in-place, and the same instance is returned.
    summary: dict
        'mislabeled': number of voxels which could keep their label (small
        parts which cannot make a vertex, or too complex geometry to allow a
        nice cut).
        'cuts': number of cuts actually performed.
        'failed_cuts': number of cut attemps which have failed.
        'changed_vertices': number of vertices which needed rework (even if
        cuts failed and nothing was actually done)
    '''

    JUNCTION = 80
    # Minimal size for splitted buckets
    min_size = 20
    # Maximal number of missed voxel to conitinue when splitting
    loc_mis_max = 50

    vs = graph['voxel_size'] #if vs is None else vs

    # Get forward and backward transformations to/from Talairach space
    tal_tr = aims.GraphManip.talairach(graph)
    tal_inv = tal_tr.inverse()

    askel = np.asarray(skel) if skel is not None else None

    # Create a new 0 filled volume
    lvol = aims.Volume(roots)
    lvol.fill(0)

    # Transform coordinates back to the native space (Talairach needed for unet)
    # TODO: allow to apply this transformation on resampled data (to speed up)
    coords = data.loc[:, ['point_x', 'point_y', 'point_z']].values
    nat_coords = tal_inv.toMatrix().dot(
        np.vstack((coords.T, np.ones(coords.shape[0]))))
    int_coords = np.round(nat_coords / np.expand_dims(vs[:] + [1.], 1)).astype(int)
    int_coords[3,:] = 0

    # Create dicts to map label's name and index
    labels = np.unique(data.after_cutting)
    labels_map, labels_rmap = {}, {}
    for i, l in enumerate(sorted(labels)):
        labels_map[l] = i + 1
        labels_rmap[i + 1] = l
    labels_rmap[0] = 'unknown'  # in case it is not already here
    # Convert labels to integers
    labels_int = np.array([labels_map[l] for l in data.after_cutting])
    # Create a volume labelled by labels index
    avol = np.asarray(lvol)
    avol[tuple(int_coords)] = labels_int

    # List the sulcus to split (those with more than one label in the bucket)
    todo = []
    for v in graph.vertices():
        apts = get_vertex_points(v)
        pts = tuple(apts)
        labels = np.unique(avol[pts])
        n_labels = len(labels)
        n_zeros = np.sum(avol[pts]==0)
        n_pts = len(pts[0])
        if n_labels == 0:
            continue
        elif n_labels == 1 or \
            (n_labels == 2 and 0 in labels and n_pts > 30 and n_zeros < 6):
            # Unique label in vertex
            label = labels[0]
            v['label'] = six.ensure_text(labels_rmap.get(label, 'unknown'))
        else:
            # Split it
            todo.append(v)

    aroots = np.asarray(roots)
    roots_new = np.max(aroots) + 1
    mislabeled = 0
    cuts = 0
    failed_cuts = 0

    changed_vertices = len(todo)

    # Split each selected vertex
    n_vert = len(list(graph.vertices()))
    while todo:
        print('{}/{} remaining bucket to split'.format(len(todo), n_vert),
              end='\r')
        v = todo.pop(0)
        apts = get_vertex_points(v)
        pts = tuple(apts)

        # Get the most represented label and the counts for each label
        winner, labels, sizes = maj_label(pts, avol, True)
        if verbose:
            print('split vertex', v.get('index'), 'in:', labels,
                  [labels_rmap.get(l, 'unknown') for l in labels],
                  ', size:', apts.shape[1], tuple(sizes))

        # Grow sub areas in the root area corresponding to this sulcus
        # Roots values (the value that define this sulcus area in root image)
        aroots = np.asarray(roots)
        roots_val = aroots[tuple(apts[:, 0])]
        # New roots values (at least 2: one per label of the sulci)
        roots_nval = list(range(roots_new, roots_new + len(labels)))

        # Update roots values (remplace roots_val by roots_nval)
        for i, l in enumerate(labels):
            lp = apts[:, np.where(avol[pts] == l)[0]]
            try:
                aroots[lp] = roots_nval[i]
            except IndexError as e:
                print('avol shape:  {}'.format(avol.shape))
                print('aroots shape:  {}'.format(aroots.shape))
                print('lp shape:  {}'.format(lp.shape))
                print(lp)
                print(np.max(lp))
                print(e)
                raise e


        # Voronoi diagram in the sulcus to extend to non classfied voxel
        # TODO: check if 6 connectivity is OK
        fm = aims.FastMarching('6')
        fm.doit(roots, [roots_val], roots_nval)
        voronoi = fm.voronoiVol()
        avor = np.asarray(voronoi)

        # Reset aroots for now, we will do it later with actual split
        # It should be done as aroots is a pointer the actual data
        for l in reversed(roots_nval):
            aroots[avor == l] = roots_val

        # List junction points in a new bucket
        split_bk = get_junction_points(apts, avor, askel, vs)

        # print('split points:', len(sb0))
        v2 = aims.FoldArgOverSegment(graph).splitVertex(v, split_bk, min_size)
        if v2 is None:
            if verbose:
                print('split failed.')
            v['label'] = six.ensure_text(labels_rmap[winner])
            failed_cuts += 1
            loc_mis = sum(sizes) - sizes[np.where(labels == winner)[0][0]]
            if verbose:
                print('mislabeled:', loc_mis)
            mislabeled += loc_mis
        else:
            cuts += 1
            apts = get_vertex_points(v)
            apts2 = get_vertex_points(v2)
            pts = tuple(apts)
            if verbose:
                print('new vertices.', len(pts[0]), apts2.shape[1])

            # Check if the new sub sulcus v1 must be splitted
            winner, labels, sizes = maj_label(pts, avol, True)
            if len(labels) != 1:
                # Number of vertex that are not equal to winner
                loc_mis = sum(sizes) - sizes[np.where(labels == winner)[0][0]]
                if verbose:
                    print('split vertex doesn\'t have one label:',
                          labels, sizes, loc_mis)
                if loc_mis > loc_mis_max:
                    todo.append(v) # do it again
                else:
                    mislabeled += loc_mis
            v['label'] = six.ensure_text(labels_rmap[winner])
            if verbose:
                print('v1 label:', v['label'])
            # new voronoi in aroots
            aroots[pts] = roots_new + 1

            # Check if the new sub sulcus v2 must be splitted
            apts = apts2
            pts = tuple(apts)
            winner, labels, sizes = maj_label(pts, avol, True)
            if len(labels) != 1:
                loc_mis = sum(sizes) - sizes[np.where(labels == winner)[0][0]]
                if verbose:
                    print('split vertex2 doesn\'t have one label:',
                          labels, sizes, loc_mis)
                if loc_mis > loc_mis_max:
                    todo.append(v2) # do it again
                else:
                    mislabeled += loc_mis
            v2['label'] = six.ensure_text(labels_rmap[winner])
            if verbose:
                print('v2 label:', v2['label'])
            aroots[pts] = roots_new

            # Expand the new labels in the sulcus
            fm = aims.FastMarching('6') # TODO: check if 6 connectivity is OK
            fm.doit(roots, [roots_val], [roots_new, roots_new + 1])
            voronoi = fm.voronoiVol()
            avor = np.asarray(voronoi)
            # change values, v gets initial roots_val, v2 gets roots_new
            aroots[avor==roots_new + 1] = roots_val

            roots_new += 1

    if verbose:
        print('mislabeled points:', mislabeled)
        print()

    summary = {
        'mislabeled': mislabeled,
        'cuts': cuts,
        'failed_cuts': failed_cuts,
        'changed_vertices': changed_vertices,
    }
    return graph, summary


if __name__ == '__main__':

    subject = 'anubis'
    center = 'panabase'
    side = 'L'
    attribs = {'subject': subject,
              'center': center,
              'side': side}
    in_file_tpl = '/neurospin/lnao/Panabase/lborne/results/sulci_recognition/database_learnclean/learning_database_learnclean/dropnet_new/left/net/result_%(side)s%(subject)s.csv'
    graph_file_tpl = '/neurospin/lnao/PClean/database_learnclean/%(center)s/%(subject)s/t1mri/t1/default_analysis/folds/3.3/base2018_manual/%(side)s%(subject)s_base2018_manual.arg'
    skel_file_tpl = '/neurospin/lnao/PClean/database_learnclean/%(center)s/%(subject)s/t1mri/t1/default_analysis/segmentation/%(side)sskeleton_%(subject)s.nii.gz'
    roots_file_tpl = '/neurospin/lnao/PClean/database_learnclean/%(center)s/%(subject)s/t1mri/t1/default_analysis/segmentation/%(side)sroots_%(subject)s.nii.gz'
    output_graph_tpl = '/tmp/%(side)s%(subject)s.arg'

    subjects_map = {
        'panabase': [
            'ammon', 'anubis', 'athena', 'atlas', 'beflo', 'caca', 'cronos',
            'demeter', 'eros', 'hades', 'horus', 'hyperion', 'icbm100T',
            'icbm125T', 'icbm200T', 'icbm201T', 'icbm300T', 'icbm310T',
            'icbm320T', 'isis', 'jah2', 'jason', 'jupiter', 'moon', 'neptune',
            'osiris', 'poseidon', 'ra', 'shiva', 'vayu', 'vishnu', 'zeus'
        ]
    }
    sides = ('L', ) #, 'R')

    elements = [{'center': center, 'subject': subject, 'side': side}
                for center, sdat in six.iteritems(subjects_map)
                for subject in sdat
                for side in sides]

    # FIXME DEBUG
    # elements = elements[:10]

    print('elements:')
    print(elements)

    mislabeled = 0
    cuts = 0
    failed_cuts = 0
    changed_vertices = 0
    summary = []

    for item in elements:
        attribs.update(item)

        print(item)

        in_file = in_file_tpl % attribs
        graph_file = graph_file_tpl % attribs
        skel_file = skel_file_tpl % attribs
        roots_file = roots_file_tpl % attribs
        output_graph = output_graph_tpl % attribs

        data = pandas.read_csv(in_file)
        graph = aims.read(graph_file)
        # skel = aims.read(skel_file)
        skel = None
        roots = aims.read(roots_file)

        graph, loc_summary = build_split_graph(graph, data, roots, skel=skel)
        summary.append(loc_summary)
        mislabeled += loc_summary['mislabeled']
        cuts += loc_summary['cuts']
        failed_cuts += loc_summary['failed_cuts']
        changed_vertices += loc_summary['changed_vertices']

        graph['label_property'] = 'label' # auto labeling
        aims.write(graph, output_graph)


    print('\nReport by subject:')
    for item, report in zip(elements, summary):
        print(item)
        print(report)
    print('\nSummary:')
    print('mislabeled:', mislabeled)
    print('changed_vertices:', changed_vertices)
    print('cuts:', cuts)
    print('failed_cuts:', failed_cuts)
