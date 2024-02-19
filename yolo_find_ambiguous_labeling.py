import os
import argparse
from typing import Tuple

def overlap_ratio(a : Tuple[int, float, float, float, float], b : Tuple[int, float, float, float, float]) -> float:
    _, a_cx, a_cy, a_w, a_h = a
    _, b_cx, b_cy, b_w, b_h = b

    a_x1 = a_cx - a_w / 2
    a_x2 = a_cx + a_w / 2
    a_y1 = a_cy - a_h / 2
    a_y2 = a_cy + a_h / 2

    b_x1 = b_cx - b_w / 2
    b_x2 = b_cx + b_w / 2
    b_y1 = b_cy - b_h / 2
    b_y2 = b_cy + b_h / 2

    i_x1 = max(a_x1, b_x1)
    i_x2 = min(a_x2, b_x2)
    i_y1 = max(a_y1, b_y1)
    i_y2 = min(a_y2, b_y2)

    area = max(0, i_x2 - i_x1) * max(0, i_y2 - i_y1)
    return area / max(a_w * a_h, b_w * b_h)


def find_ambiguous_labelings(filepath : str, threshold : float = 0.5):
    objects = list()

    with open(filepath, "r") as file:
        for line in file.readlines():
            line = line.strip()
            if line == "":
                continue

            label, cx, cy, width, height = line.split(' ')
            objects.append((
                int(label),
                float(cx),
                float(cy),
                float(width),
                float(height)
            ))
    
    for i, a in enumerate(objects):
        for j in range(i+1, len(objects)):
            b = objects[j]

            if overlap_ratio(a, b) > threshold:
                print("    {} {} {} {} {} <-> {} {} {} {} {}".format(*a, *b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str)
    args = ap.parse_args()
    
    basedir = os.path.join(args.src, "obj_train_data")
    for filename in os.listdir(basedir):
        _, ext = os.path.splitext(filename)
        if ext != ".txt":
            continue

        print(filename)
        find_ambiguous_labelings(os.path.join(basedir, filename))

if __name__ == "__main__":
    main()
