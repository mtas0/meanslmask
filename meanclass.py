import glob
from os.path import join as opj
import nibabel as nib
from nilearn.image import mean_img


class GetMeanSL:
    """Class to get mean superimposed searchlight masks.
    To optimize: use better mask and task function so self.mask and task
    are not dependent on previous string"""

    def __init__(self, task, mask):
        self.task = task
        self.mask = mask

    def _get_mask(self, regmask_dir):
        """get specific mask"""
        return rf"{regmask_dir}\binarized_{self.mask}.nii.gz"

    def get_task_ni(self, clf_dir, task):
        """returns all numpy or nii files per task
        for numpy type npy for nii files type nii.gz"""
        return glob.glob(opj(clf_dir, rf"{task}\sub-*\*.nii.gz"))

    def impose_mask(self, mask, file):
        """multiplying data for superimposing on mask"""
        m = nib.load(mask)
        print(m)
        affine = file.affine
        mult = file.get_fdata() * m.get_fdata()
        # print(mult.min(), mult.max())
        return nib.Nifti1Image(mult, affine)

    def __call__(self, clf_dir, regmask_dir):
        mask = self._get_mask(regmask_dir)
        fil = self.get_task_ni(clf_dir, self.task)
        return self.impose_mask(mask, mean_img(fil))
