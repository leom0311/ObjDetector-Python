import os
import argparse
from typing import Tuple
import matplotlib.pyplot as P
import random

from .yolo_dataset import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str)
    args = ap.parse_args()

    class_names = read_class_list(os.path.join(args.src, "obj.names"))
    population = [0 for _ in class_names]
    
    with open(os.path.join(args.src, "train.txt"), "r") as list_file:
        for line in list_file.readlines():
            line = line.strip()
            if line == "":
                continue

            if not line.startswith("data/"):
                raise Exception("expected 'data/' at the beginning of each path")
            
            *dirnames, filename = line[5:].split('/')
            stem, _ = os.path.splitext(filename)

            obj_list = read_object_list(os.path.join(args.src, *dirnames, stem + ".txt"))

            for obj in obj_list:
                population[obj.class_id] += 1
    
    for count, name in zip(population, class_names):
        print(name, count, sep = '\t')

if __name__ == "__main__":
    main()
