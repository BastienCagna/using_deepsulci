import numpy as np
from soma import aims, aimsalgo
from scipy.spatial.distance import dice


def avg_min_euclidean_dist(a, b):
    """ Euclidean distance between two vector (element by element) """
    diff = np.tile(b, (a.shape[0], 1)) - np.repeat(a, b.shape[0], axis=0)
    dist_vector = np.sqrt(np.sum(np.power(diff, 2), axis=1))
    return np.mean(np.min(dist_vector.reshape(a.shape[0], b.shape[0]), axis=1))


def avg_min_euclidean_dist_job(bck_a, buckets):
    return list(avg_min_euclidean_dist_job(bck_a, bck_b) for bck_b in buckets)


def graphs_similarity_matrix(graphs):
    n_graphs = len(graphs)
    buckets = (dtb.graph.list_buckets(g, transform="Talairach")[0]
               for g in graphs)
    matrix = np.zeros((n_graphs, n_graphs))
    # for i, gi_buckets in enumerate(buckets):
    #     for j, gj_buckets in enumerate(buckets):
    #         if i != j:
    #             for bck_a in gi_buckets:
    #                 matrix[i, j] += min(avg_min_euclidean_dist(bck_a, bck_b) for bck_b in gj_buckets)
    #             matrix[i, j] /= len(gj_buckets)

    for i, gi_buckets in enumerate(buckets):
        for j, gj_buckets in enumerate(buckets):
            if i != j:
                dists = []
                for bck_a in gi_buckets:
                    print(list(bck_a))
                #     dists.append(
                #         avg_min_euclidean_dist_job(bck_a, gj_buckets))
                # # dists = Parallel(n_jobs=max(cpu_count() - 2, 1))(
                # #     delayed(avg_min_euclidean_dist_job)(bck_a, gj_buckets) for bck_a in gi_buckets)
                # dists = dists.flatten()
                # for a, bck_a in enumerate(gi_buckets):
                #     for b, bck_b in enumerate(gj_buckets):
                #         matrix[i, j] += min(dists[a*len(gj_buckets) + b])
                # matrix[i, j] /= len(gj_buckets)

    return matrix


def transform(img_f, trm_f):
    vol = aims.read(img_f)
    transform = aims.read(trm_f)
    rf = getattr(aims, 'ResamplerFactory_%s' % aims.voxelTypeCode(vol))()
    print(aims.voxelTypeCode(vol))
    resampler = rf.getResampler(0)
    # allouer un volums destination avec la taille voulue
    out_vol = aims.Volume(100, 100, 100, dtype=aims.voxelTypeCode(vol))
    out_vol.header()['voxel_size'] = [2., 2., 2.]
    resampler.resample(vol, transform, 0, out_vol)  # 0: background value
    return out_vol


def dice_similarity_matrix(image_files, trm_files, label=1):

    # Load all images in the Talairach space
    images = []
    for img_f, trm_f in zip(image_files, trm_files):
        img = transform(img_f, trm_f)

        # resampler = aims.ResamplerFactory_S16().getResampler(0)
        # resampler.setDefaultValue(0)  # set background to -1
        # resampler.setRef(img)  # volume to resample
        # resampled = resampler.doit(trm_f, 200, 200, 200, (2., 2., 2.))

        images.append(img)  # np.array(resampled).flatten())

    n_graphs = len(images)
    matrix = np.zeros((n_graphs, n_graphs))
    for i, img_a in enumerate(images):
        for j, img_b in enumerate(images[i:]):
            d = dice(img_a, img_b)
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


if __name__ == "__main__":
    s = dice_similarity_matrix(
        [
            "/neurospin/dico/data/bv_databases/human/pclean/all/beflo/t1mri/t1/default_analysis/segmentation/Lgrey_white_beflo.nii.gz",
            "/neurospin/dico/data/bv_databases/human/pclean/all/s12913/t1mri/t1/default_analysis/segmentation/Lgrey_white_s12913.nii.gz",
            "/neurospin/dico/data/bv_databases/human/pclean/all/sujet04/t1mri/t1/default_analysis/segmentation/Lgrey_white_sujet04.nii.gz",
            "/neurospin/dico/data/bv_databases/human/pclean/all/icbm310T/t1mri/t1/default_analysis/segmentation/Lgrey_white_icbm310T.nii.gz",
        ],
        [
            "/neurospin/dico/data/bv_databases/human/pclean/all/beflo/t1mri/t1/default_analysis/segmentation/Lgrey_white_beflo.nii.gz",
            "/neurospin/dico/data/bv_databases/human/pclean/all/s12913/t1mri/t1/default_analysis/segmentation/Lgrey_white_s12913.nii.gz",
            "/neurospin/dico/data/bv_databases/human/pclean/all/sujet04/t1mri/t1/default_analysis/segmentation/Lgrey_white_sujet04.nii.gz",
            "/neurospin/dico/data/bv_databases/human/pclean/all/icbm310T/t1mri/t1/default_analysis/segmentation/Lgrey_white_icbm310T.nii.gz",
        ]
    )
