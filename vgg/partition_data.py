from typing import List
import os
from shutil import copyfile
from pathlib import Path


def copy_files(in_list: List[Path], out_path: Path):
    """
    Copy all files in in_list to out_path

    Parameters
    ----------
    in_list: List[Path]
        List of files to copy
    out_path: Path
        Destination
    """
    for f in in_list:
        copyfile(f, Path(out_path, f.name))


def clean_folders(folders: List[Path]):
    """
    Remove content all folders in input list.

    Patameters
    ----------
    folders: List[Path]
        list of folders
    """
    for f in folders:
        files = f.glob("*")
        if len(files) == 0:
            continue
        for ff in files:
            os.remove(ff)


def partition_data(input_path: Path, output_path: Path, split=0.7):
    """
    Split data and place in folders used to create generators from

    Parameters
    ----------
    input_path: Path
        path to "unsplit" data
    output_path: Path
        path to write split data to
    split: float
        split fraction (fraction of train data)
    """
    # List all files
    maz = list(Path(input_path, "maz").glob("*.jpg"))
    rey = list(Path(input_path, "rey").glob("*.jpg"))

    # Split files into classes and train/test set
    maz_split = int(split * len(maz))
    maz_train, maz_eval = maz[:maz_split], maz[maz_split:]

    rey_split = int(split * len(rey))
    rey_train, rey_eval = rey[:rey_split], rey[rey_split:]

    # Paths to output folders
    maz_train_folder = Path(output_path, "train", "maz")
    maz_valid_folder = Path(output_path, "valid", "maz")
    rey_train_folder = Path(output_path, "train", "rey")
    rey_valid_folder = Path(output_path, "valid", "rey")

    # Remove content of folders
    clean_folders([maz_train_folder, maz_valid_folder,
                   rey_train_folder, rey_valid_folder])

    # copy images to folders
    copy_files(maz_train, maz_train_folder)
    copy_files(maz_eval, maz_valid_folder)

    copy_files(rey_train, rey_train_folder)
    copy_files(rey_eval, rey_valid_folder)


if __name__ == "__main__":
    in_path = Path("data", "raw")
    train_path = Path("data", "processed")
    partition_data(in_path, train_path)
