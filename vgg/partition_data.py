from typing import List
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


def partition_data(input_path: Path, output_path: Path, split=0.9):
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

    # copy images to folders
    copy_files(maz_train, Path(output_path, "train", "maz"))
    copy_files(maz_eval, Path(output_path, "valid", "maz"))

    copy_files(rey_train, Path(output_path, "train", "rey"))
    copy_files(rey_eval, Path(output_path, "valid", "rey"))


if __name__ == "__main__":
    in_path = Path("data", "raw")
    train_path = Path("data", "processed")
    partition_data(in_path, train_path)
