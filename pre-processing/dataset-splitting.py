import os
import pandas as pd
    import glob
from sklearn.model_selection import train_test_split


def get_mask_path():
    return os.getcwd() + "/../IXI-masks"

def get_ixi_path():
    return os.getcwd() + "/../IXI-T1"

def get_metainfo_path():
    return os.getcwd() + "/../IXI-T1/IXI.xls"

def get_final_set_path():
    return os.getcwd() + "/../final_set"

def get_png_path():
    return os.getcwd() + "/../PNGs/images"

def generate_subsets():
    df = pd.read_excel(get_metainfo_path())
    df = df.dropna()

    # First we need to convert age column to a new categorical one based on range
    bins = [0, 40, 65, 120]
    labels = ['Young', 'Mature', 'Elder']
    df['AgeGroup'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

    # Most subjects are from white (1) ethnicity. The other groups have little representation or are bad classified,
    # which could lead to problems with stratified sampling, so we group them all in the Others (6) ethnicity
    df.loc[df['ETHNIC_ID'] == 2, 'ETHNIC_ID'] = 6
    df.loc[df['ETHNIC_ID'] == 0, 'ETHNIC_ID'] = 6
    df.loc[df['ETHNIC_ID'] == 3, 'ETHNIC_ID'] = 6
    df.loc[df['ETHNIC_ID'] == 4, 'ETHNIC_ID'] = 6
    df.loc[df['ETHNIC_ID'] == 5, 'ETHNIC_ID'] = 6

    # The stratified sampling, using sklearn tain/test split
    df_train_val, df_test = train_test_split(df, test_size=0.1, stratify=df[["AgeGroup", "ETHNIC_ID", "SEX_ID (1=m, 2=f)"]])

    # Dividing into validation on train is done in this stage, since ImageDataGenerator methods, suchs a FlowFromDirectory,
    # would lead to not stratified sampling and could mix slices from the same subjects in different sets
    df_train, df_val = train_test_split(df_train_val, test_size=0.2, stratify=df_train_val[["AgeGroup", "ETHNIC_ID", "SEX_ID (1=m, 2=f)"]])

    # Now we have the subjects in different dataframes (test, train , validation) and slices in a common folder belonging
    # to the different original NIFTI volume.
    dest_dir = get_final_set_path()
    src_dir = get_png_path()
    for i, row in df_train.iterrows():
        ixi_id = row['IXI_ID']
        move_subject_files(ixi_id, f'{src_dir}/full', f'{dest_dir}/full/train')
        move_subject_files(ixi_id, f'{src_dir}/skull-stripped', f'{dest_dir}/skull-stripped/train')

    for i, row in df_val.iterrows():
        ixi_id = row['IXI_ID']
        move_subject_files(ixi_id, f'{src_dir}/full', f'{dest_dir}/full/val')
        move_subject_files(ixi_id, f'{src_dir}/skull-stripped', f'{dest_dir}/skull-stripped/val')

    for i, row in df_test.iterrows():
        ixi_id = row['IXI_ID']
        move_subject_files(ixi_id, f'{src_dir}/full', f'{dest_dir}/full/test')
        move_subject_files(ixi_id, f'{src_dir}/skull-stripped', f'{dest_dir}/skull-stripped/test')

    # The remaining files in src dirs go to test subset and were manually moved


def move_subject_files(subject_id, src_dir, dest_dir):
    # Get subject slices as PNG and move to the new folder
    # Full skulls
    files = glob.glob(f'{src_dir}/IXI{str(subject_id).zfill(3)}*.png')
    for file in files:
        dest_file = f'{dest_dir}/{os.path.basename(file)}'
        os.rename(file, dest_file)


# This is a regular python script to get a dataset with train, test and validation subesets as different folder
# It uses stratified sampling to ensure well-balanced sets according to different aspects in the data based on the
# IXI metainfo info file IXI.xls
if __name__ == '__main__':
    generate_subsets()

