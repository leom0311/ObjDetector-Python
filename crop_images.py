import os
import argparse
from PIL import Image
import torchvision.transforms.functional as TF

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str)
    ap.add_argument("-out", type = str)
    ap.add_argument("-region", type = int, nargs = 4)
    ap.add_argument("-extensions", type = str, nargs = "+", default = [".bmp", ".jpg", ".png"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok = True)

    left, top, width, height = args.region

    for filename in os.listdir(args.src):
        if not os.path.isfile(os.path.join(args.src, filename)):
            continue

        stem, ext = os.path.splitext(filename)
        if ext not in args.extensions:
            continue

        print(filename)

        image = Image.open(os.path.join(args.src, filename))
        image = TF.crop(image, top, left, height, width)
        image.save(os.path.join(args.out, stem + ".png"))


if __name__ == "__main__":
    main()
