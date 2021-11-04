"""

"""

# Author : Bastien Cagna (bastiencagna@gmail.com)

import os.path as op
from os import makedirs, listdir
from joblib import Parallel, cpu_count, delayed
import numpy as np
import json
import sys
import argparse

from soma import aims, aimsalgo
from using_deepsulci.cohort import Cohort, SubjectDataset
from resample_labeled_volume import resample
from utils import real_njobs


def resampled_graph(graph_f, output_vs, out_dir, suffix="_resampled"):
    if not graph_f:
        return None
    fname = op.split(graph_f)[1][:-4] + suffix + ".arg"
    fpath = op.join(out_dir, fname)

    trm = aims.AffineTransformation3d(np.eye(4))
    inv_trm = trm.inverse()

    if not op.exists(fpath):
        print("Resampling to " + fpath)
        graph = aims.read(graph_f)
        aimsalgo.transformGraph(graph, trm, inv_trm, output_vs)
        aims.write(graph, fpath)
    return fpath


def resample_volume(in_file, output_vs, out_dir, suffix="_resampled", order=1):
    if not in_file:
        return None
    fname = op.split(in_file)[1][:-7] + suffix + ".nii.gz"
    fpath = op.join(out_dir, fname)

    trm = aims.AffineTransformation3d(np.eye(4))
    inv_trm = trm.inverse()

    if not op.exists(fpath):
        print("Resampling to " + fpath)
        vol = aims.read(in_file)

        # New volume dimensions
        resampling_ratio = np.array(vol.header()['voxel_size'][:3]) / output_vs
        orig_dim = vol.header()['volume_dimension'][:3]
        new_dim = list((resampling_ratio * orig_dim).astype(int))

        resampler = aimsalgo.ResamplerFactory(vol).getResampler(order)
        resampler.setRef(vol)
        resampled = aims.Volume(new_dim, dtype=vol.np.dtype)
        resampled.header()['voxel_size'] = output_vs
        resampler.resample_inv(vol, inv_trm, 0, resampled)

        aims.write(resampled, fpath)
    return fpath


def resample_labeled(in_file, output_vs, out_dir, suffix="_resampled"):
    if not in_file:
        return None
    fname = op.split(in_file)[1][:-7] + suffix + ".nii.gz"
    fpath = op.join(out_dir, fname)

    if not op.exists(fpath):
        print("Resampling to " + fpath)
        resampled = resample(in_file, None, output_vs)
        aims.write(resampled, fpath)
    return fpath


def resample_subject_job(sub, vx_size, rs_dir, suffix):
    rs_dir = op.join(rs_dir, sub.name)
    makedirs(rs_dir, exist_ok=True)
    return SubjectDataset(
        sub.name,
        resample_volume(sub.t1, vx_size, rs_dir, suffix),
        resample_labeled(sub.roots, vx_size, rs_dir, suffix),
        resample_labeled(sub.skeleton, vx_size, rs_dir, suffix),
        resampled_graph(sub.graph, vx_size, rs_dir, suffix),
        resampled_graph(sub.notcut_graph, vx_size, rs_dir, suffix),
    )


def add_resampled_graphs(cohort_f, rs_dir, vx_size, suffix, n_jobs=1):
    cohort = Cohort(from_json=cohort_f)
    print("Resampling cohort: " + cohort.name)
    subjects = Parallel(n_jobs=real_njobs(n_jobs))(delayed(
        resample_subject_job)(sub, vx_size, rs_dir, suffix)
        for sub in cohort.subjects)

    rs_cohort = Cohort(cohort.name + suffix, subjects)
    cohort_dir, _ = op.split(cohort_f)
    rs_cohort.to_json(op.join(cohort_dir, "cohort-" + rs_cohort.name + ".json"))


def main():
    parser = argparse.ArgumentParser(description='Resample cohort files to desired resolution')
    parser.add_argument('-c', dest='cohorts', type=str, nargs='+', default=None, required=False,
                        help='Cohort names')
    parser.add_argument('-sx', type=float, default=2, help="Voxel size on X axis")
    parser.add_argument('-sy', type=float, default=2, help="Voxel size on Y axis")
    parser.add_argument('-sz', type=float, default=2, help="Voxel size on Z axis")
    parser.add_argument('-n', dest='njobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('-e', dest='env', type=str, default=None, help="Configuration file")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    # Resample to 2mm iso
    output_vs = (args.sx, args.sy, args.sz)
    rsdir = op.join(env['working_path'], "resampled")
    makedirs(rsdir, exist_ok=True)

    # Set cohorts paths
    chrt_dir = op.join(env['working_path'], "cohorts")
    if args.cohorts:
        cohorts_files = []
        for cname in args.cohorts:
            cohorts_files.append(
                op.join(chrt_dir, "cohort-" + cname + ".json")
            )
    else:
        cohorts_files = list(op.join(chrt_dir, f) for f in listdir(chrt_dir))

    if output_vs[0] == output_vs[1] and output_vs[0] == output_vs[2]:
        suffix = "_resampled-{}mm".format(output_vs[0])
    else:
        suffix = "_resampled-{}x{}x{}mm".format(output_vs[0], output_vs[1], output_vs[2])

    for cfile in cohorts_files:
        if cfile[-7:-5] != "mm":
            add_resampled_graphs(cfile, rsdir, output_vs, suffix, args.njobs)


if __name__ == "__main__":
    main()
