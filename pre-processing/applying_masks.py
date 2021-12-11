import os
import sys
import matplotlib.image
import nibabel as nib
from nibabel.testing import data_path
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import cluster
import cv2


def get_ds_path():
    return os.getcwd() + "/../IXI-T1"

def get_mask_path():
    return os.getcwd() + "/../IXI-masks"


def get_filtered_mask_path():
    return os.getcwd() + "/../IXI-filtered-masks"


def get_all_files():
    path = get_ds_path()
    # files = glob.glob(f'{path}/*.nii.gz')
    files = glob.glob(f'{path}/IXI002-Guys-0828-T1.nii.gz')
    return files

def apply_masks_test():
    files = get_all_files()
    apply_mask(files[0])
    # for i in range(100, 105):
    #     apply_mask(files[i])

def mask_file_by_nii_gz(nii_name):
    # path = get_mask_path()
    path = get_filtered_mask_path()
    nii_name = str.replace(nii_name, '-T1.nii.gz', '')
    mask_files = glob.glob(f'{path}/{nii_name}*.nii.mask.npy')
    return mask_files[0]

def apply_mask(file):
    slice_pos = 74  # 74 is the middle of the MRI

    # Load volume as numpy
    img = nib.load(file).get_fdata()

    # Resolve mask file path
    mask_file_name = mask_file_by_nii_gz(os.path.basename(file))

    # Load the mask as numpy
    # mask = np.load(os.path.join(get_mask_path(), mask_file_name))
    mask = np.load(os.path.join(get_filtered_mask_path(), mask_file_name))

    # Apply the mask to the volume
    volume_after_mask = np.multiply(img, mask)

    # Explore the results (show some slices)
    # plot the mask
    mask_slice = np.squeeze(mask[:, :, slice_pos:slice_pos+1]) # Middle slice

    # plt.imshow(mask_slice.T, cmap="gray", origin="lower")
    # plt.show()

    # plot the filtered slice
    slice = np.squeeze(volume_after_mask[:, :, slice_pos:slice_pos+1])
    plt.imshow(slice.T, cmap="gray", origin="lower")
    plt.show()

    # plot the non-filtered slice:
    slice = np.squeeze(img[:, :, slice_pos:slice_pos+1])
    plt.imshow(slice.T, cmap="gray", origin="lower")
    plt.show()

    cluster_mask(mask_file_name)



def cluster_mask(mask_file):
    slice_pos = 74

    # mask_file = get_mask_path() + '/IXI661-HH-2788-MADisoTFE1_-s3T253_-0301-00003-000001-01.nii.mask.npy'
    mask = np.load(mask_file)
    mask_slice = np.squeeze(mask[:, :, slice_pos:slice_pos+1])
    # plt.imshow(mask_slice.T, cmap="gray", origin="lower")
    # plt.show()

    # transform to x,y coordinates
    tmask = []
    for i in range(0,255):
        for j in range(0, 255):
            if mask_slice[i,j]:
                tmask.append([i,j])

    tmask = np.array(tmask)

    eps = 10
    min_samples = 1
    dbscanModel = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    ygrup = dbscanModel.fit(tmask)
    labels = ygrup.labels_
    plt.scatter(tmask[:,0], tmask[:,1], c=labels, alpha=.3, cmap='jet')
    plt.show()

    # We know that the cluster/label with more instances is the one we want to preserve
    unique, counts = np.unique(labels, return_counts=True)
    selected_label = unique[np.argmax(counts)]

    # Final mask
    n = 0
    for i in range(0,255):
        for j in range(0, 255):
            if mask_slice[i,j]:
                mask_slice[i,j] = labels[n] == selected_label
                n += 1

    plt.imshow(mask_slice.T, cmap="gray", origin="lower")
    plt.show()


# This is a regular python script for testing deepbrain masks applied to NIFTI MRIs
if __name__ == '__main__':
    apply_masks_test()
    # cluster_mask()

