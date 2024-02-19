import os
import argparse
from typing import Tuple
from PIL import Image, ImageDraw
import matplotlib.pyplot as P
import random
import torchvision.transforms.functional as T

from .yolo_dataset import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str)
    ap.add_argument("-cls", type = str)
    args = ap.parse_args()

    class_names = read_class_list(os.path.join(args.src, "obj.names"))
    class_id = class_names.index(args.cls)

    found_objects : List[Tuple[str, AnnotatedObject]] = list()
    
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
            image_path = os.path.join(args.src, *dirnames, filename)

            for obj in obj_list:
                if obj.class_id == class_id:
                    found_objects.append(( image_path, obj ))
    
    random.shuffle(found_objects)

    for image_path, obj in found_objects:
        image = Image.open(image_path)
        x1 = image.width  * (obj.cx - obj.width )
        x2 = image.width  * (obj.cx + obj.width )
        y1 = image.height * (obj.cy - obj.height)
        y2 = image.height * (obj.cy + obj.height)

        image = T.resized_crop(image, top = round(y1), left = round(x1), height = round(y2) - round(y1), width = round(x2) - round(x1), size = [128, 128])
        draw = ImageDraw.Draw(image)
        draw.rectangle((32, 32, 96, 96), outline = "red")

        P.imshow(image)
        P.show()

if __name__ == "__main__":
    main()
