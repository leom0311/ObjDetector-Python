import os
import argparse
import shutil
from PIL import Image
import torchvision.transforms.functional as T

def crop_images_and_annotations(srcdir : str, destdir : str, crop_left : int, crop_top : int, crop_width : int, crop_height : int):
    with open(os.path.join(srcdir, "train.txt"), "r") as src_list_file:
        with open(os.path.join(destdir, "train.txt"), "w") as dest_list_file:
            for image_relpath in src_list_file.readlines():
                image_relpath = image_relpath.strip()
                if image_relpath == "":
                    continue

                print(image_relpath, file = dest_list_file)

                dir1, *dirnames, filename = image_relpath.split('/')
                if dir1 != "data":
                    raise Exception("Expected first part of relative path to be 'data'.")
                
                # crop image
                image = Image.open(os.path.join(srcdir, *dirnames, filename))
                original_width  = image.width
                original_height = image.height

                image = T.crop(image, crop_top, crop_left, crop_height, crop_width)
                
                os.makedirs(os.path.join(destdir, *dirnames), exist_ok = True)
                image.save(os.path.join(destdir, *dirnames, filename))

                # update annotations
                stem, _ = os.path.splitext(filename)

                with open(os.path.join(srcdir, *dirnames, stem + ".txt"), "r") as src_annotations_file:
                    with open(os.path.join(destdir, *dirnames, stem + ".txt"), "w") as dest_annotations_file:
                        for annotation in src_annotations_file.readlines():
                            annotation = annotation.strip()
                            if annotation == "":
                                continue

                            clsid, cx, cy, w, h = annotation.split(' ')
                            cx = float(cx)
                            cy = float(cy)
                            w  = float(w)
                            h  = float(h)

                            cx = (cx * original_width  - crop_left) / crop_width
                            cy = (cy * original_height - crop_top ) / crop_height
                            w  = w  * original_width  / crop_width
                            h  = h  * original_height / crop_height

                            if cx + w/2 <= 0 or cx - w/2 >= 1 or cy + h/2 <= 0 or cy - h/2 >= 1:
                                continue

                            print("{} {} {} {} {}".format(clsid, cx, cy, w, h), file = dest_annotations_file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str)
    ap.add_argument("-dest", type = str)
    ap.add_argument("-region", type = int, nargs = 4)
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok = True)

    left, top, width, height = args.region
    crop_images_and_annotations(args.src, args.dest, left, top, width, height)

    shutil.copyfile(os.path.join(args.src, "obj.data"), os.path.join(args.dest, "obj.data"))
    shutil.copyfile(os.path.join(args.src, "obj.names"), os.path.join(args.dest, "obj.names"))
