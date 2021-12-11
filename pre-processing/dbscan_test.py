import os
import sys
import matplotlib.image
import nibabel as nib
from nibabel.testing import data_path
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets
import cv2


def get_ds_path():
    return os.getcwd() + "/../IXI-T1"

def get_mask_path():
    return os.getcwd() + "/../IXI-masks"

def get_all_files():
    path = get_ds_path()
    files = glob.glob(f'{path}/*.nii.gz')
    return files

def mask_file_by_nii_gz(nii_name):
    path = get_mask_path()
    nii_name = str.replace(nii_name, '-T1.nii.gz', '')
    mask_files = glob.glob(f'{path}/{nii_name}*.nii.mask.npy')
    return mask_files[0]

def dbscan_test():
    files = get_all_files()
    # Only interested in one instance to begin with
    file = files[0]

    # 1. First we load a MRI and applied the mask to it
    img = nib.load(file).get_fdata()
    mask_file_name = mask_file_by_nii_gz(os.path.basename(file))
    mask = np.load(os.path.join(get_mask_path(), mask_file_name))
    volume_after_mask = np.multiply(img, mask)

    # 2. Once we have the filtered volume, it's time to use dbscan
    # eps = 0.5
    # min_samples = 20
    # dbscan_model = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    # ygrup = dbscan_model.fit(volume_after_mask)

    # !! Throws an error that suggest we should work on 2D
    # ValueError: Found   array   with dim 3. Estimator expected <= 2.
    # Options:
    # 1. reshape: https://stackoverflow.com/questions/34972142/sklearn-logistic-regression-valueerror-found-array-with-dim-3-estimator-expec
    # 2. Work wit slices <----

    # Working with slices
    slice = np.squeeze(img[:, :, 74:75])
    eps = 0.5
    min_samples = 20
    dbscan_model = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    ygrup = dbscan_model.fit(slice)

    # fig, ax = plt.subplots(3, 3, figsize=(13, 10))
    plt.scatter(slice[:,0], slice[:,1], c=ygrup.labels_, alpha=.3, cmap='jet')
    plt.show()
    # plt.set_title(f"dbscan eps={eps}, min_samples={min_samples}")

def basic_test():
    X_blobs, y_blobs = datasets.make_blobs(n_samples=1000, n_features=2, centers=4, cluster_std=1.6, random_state=42)
    X, y = X_blobs, y_blobs
    eps = 0.5
    min_samples = 10
    dbscanModel = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    ygrup = dbscanModel.fit(X)
    plt.scatter(X[:,0], X[:,1], c=ygrup.labels_, alpha=.3, cmap='jet')
    plt.show()

def basic_test2():
    img = cv2.imread("slice.png")
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    n = 0
    while (n < 4):
        labimg = cv2.pyrDown(labimg)
        n = n + 1

    feature_image = np.reshape(labimg, [-1, 3])
    rows, cols, chs = labimg.shape

    db = cluster.DBSCAN(eps=15, min_samples=50, metric='euclidean', algorithm='auto')
    db.fit(feature_image)
    labels = db.labels_

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(np.reshape(labels, [rows, cols]))
    plt.axis('off')
    plt.show()



# This python file starts from the applying_masks.py testing script. In that script, whe showed
# how deepbrain does a good job creating a mask for identifying only the brain tisues from skull MRIs
# Wa want now to explore whether it is possible to use dbscan to filter out some false positive voxels that appeared
# in the aforementioned step
if __name__ == '__main__':
    # basic_test2()
    basic_test()
    # dbscan_test()
