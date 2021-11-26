# -*- coding: utf-8 -*-

import os
import zipfile
import tqdm
import pandas as pd


def unzip(zip_path, unzip_path=None):
    """unzip

    Args:
        zip_path (str): zip path
        unzip_path (str, optional): unzip path. Defaults to None.

    Returns:
        unzip_path: unzip path
    """

    # set unzip path
    if unzip_path == None:
        unzip_path = os.path.splitext(zip_path)[0]

    # unzip
    with zipfile.ZipFile(zip_path) as zf:
        for file in tqdm.tqdm(zf.infolist(), desc="unzip..."):
            try:
                zf.extract(file, unzip_path)
            except zipfile.error as e:
                print(e)

    return unzip_path


def save_to_csv(path, data, name):
    """[summary]

    Args:
        path (str): save path
        data (dict): pandas data input
        name (str): csv file name
    """
    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(path, name), index=False)
