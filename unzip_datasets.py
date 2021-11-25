# -*- coding: utf-8 -*-

import os
import toml
from zipfile import ZipFile


def unzip_dataset(path):
    extract_dir = os.path.splitext(path)[0]
    fp = ZipFile(path, "r")
    fp.extractall(extract_dir)
    return extract_dir


if __name__ == "__main__":
    # get config
    toml_path = os.path.join(os.getcwd(), "config", "unzip_datasets_cfg.toml")
    config = toml.load(toml_path)

    # get datasets zip path
    root_path = os.path.abspath(config["path"]["root"])
    zip_path = os.path.join(root_path, config["path"]["zip"])

    # extract datasets zip
    datasets_path = os.path.splitext(zip_path)[0]
    if not os.path.exists(datasets_path):
        print("extract ing......")
        datasets_path = unzip_dataset(zip_path)
    print(f"datasets path {datasets_path}")
