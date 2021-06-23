import traits.api as traits
import numpy as np
from soma import aims, aimsalgo
from capsul.api import Process


class LabelResample(Process):
    '''

    '''

    def __init__(self):
        super(LabelResample, self).__init__()

        self.add_trait('input_image', traits.File(
            output=False, desc='Labelled image to transform'))
        self.add_trait('transformation', traits.File(
            output=False, optional=True, desc='Transformation file .trm'))
        self.add_trait('sx', traits.Float(-1, output=False,
                                        desc='Output resolution (X axis)'))
        self.add_trait('sy', traits.Float(-1, output=False,
                                        desc='Output resolution (Y axis)'))
        self.add_trait('sz', traits.Float(-1, output=False,
                                        desc='Output resolution (Z axis)'))
        self.add_trait('background', traits.Int(0, output=False,
                                        desc='Background value/label'))
        self.add_trait('output_image', traits.File(
            output=False, desc='file (.json) storing the hyperparameters'
                               ' (cutting threshold)'))

    def _run_process(self):
        # Read inputs
        vol = aims.read(self.input_image)

        if self.transformation:
            trm = aims.read(self.transformation)
        else:
            trm = aims.AffineTransformation3d(np.eye(4))
        inv_trm = trm.inverse()

        if self.sx > 0:
            output_vs = (self.sx, self.sy, self.sz)
        else:
            output_vs = vol.header()['voxel_size'][:3]

        # Transform the background
        # Using the inverse is more straightforward and supports non-linear
        # transforms
        # FIXME: keep fixed size look weird. Should use col.header() ?
        resampled = aims.Volume((100, 120, 100), dtype=vol.__array__().dtype)
        resampled.header()['voxel_size'] = output_vs
        # 0 order (nearest neightbours) resampling
        resampler = aimsalgo.ResamplerFactory(vol).getResampler(0)
        resampler.setDefaultValue(self.background)
        resampler.setRef(vol)
        resampler.resample_inv(vol, inv_trm, 0, resampled)

        # # Create buckets
        # bck = aims.BucketMap_VOID()
        # bck.setSizeXYZT(*vol.header()['voxel_size'][:3], 1.)
        # # build a single bucket from the volume values where voxel are non
        # # equal to background value
        # bk0 = bck[0]
        # # TODO: This takes lot of times ==> parrallelize?
        # for p in np.vstack(np.where(vol.__array__() != self.background)[:3]).T:
        #     bk0[list(p)] = 1
        #
        # # Transform buckets
        # # /!\ this function has a bug for aims <= 5.0.1
        # bck2 = aimsalgo.resampleBucket(bck, trm, inv_trm, output_vs)
        # # bck2 = aimsalgo.transformBucketDirect(bck, tr, output_vs)

        # Rebuild image from buckets
        # conv = aims.Converter(intype=bck2, outtype=aims.AimsData(vol))
        # conv.convert(bck2, resampled)

        # Merge images

        aims.write(resampled, self.output_image)
