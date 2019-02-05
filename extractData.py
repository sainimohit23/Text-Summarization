import os
import tarfile
import gzip
import zipfile
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--glove", action="store_true")
args = parser.parse_args()

# Extract data file
with tarfile.open("summary.tar.gz", "r:gz") as tar:
    tar.extractall()

with gzip.open("sumdata/train/train.article.txt.gz", "rb") as gz:
    with open("sumdata/train/train.article.txt", "wb") as out:
        out.write(gz.read())

with gzip.open("sumdata/train/train.title.txt.gz", "rb") as gz:
    with open("sumdata/train/train.title.txt", "wb") as out:
        out.write(gz.read())