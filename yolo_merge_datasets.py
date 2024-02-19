import os
import argparse
import shutil
from typing import List

def merge_datasets(srcdirs : List[str], destdir : str):
    data_dir_name = "obj_train_data"

    os.makedirs(destdir, exist_ok = True)
    os.makedirs(os.path.join(destdir, data_dir_name), exist_ok = True)

    obj_names       = []
    name_to_id      = dict()
    next_sample_id  = 0

    with open(os.path.join(destdir, "train.txt"), "w") as dest_list_file:
        for srcdir in srcdirs:
            # read class names
            src_obj_names = []
            with open(os.path.join(srcdir, "obj.names"), "r") as names_file:
                for line in names_file.readlines():
                    line = line.strip()
                    if line == "":
                        continue
                    src_obj_names.append(line)
            
            # update class list
            for i, name in enumerate(src_obj_names):
                if name not in name_to_id:
                    name_to_id[name] = len(obj_names)
                    obj_names.append(name)
            
            # copy samples
            with open(os.path.join(srcdir, "train.txt"), "r") as src_list_file:
                for relpath in src_list_file.readlines():
                    relpath = relpath.strip()
                    if relpath == "":
                        continue

                    dir1, *dirnames, filename = relpath.split('/')
                    if dir1 != "data":
                        raise Exception("Expected 'data' as first part of relative path")
                    
                    filename_base, filename_ext = os.path.splitext(filename)
                    
                    dest_sample_name = "{:08d}".format(next_sample_id)
                    next_sample_id += 1

                    # copy image file
                    shutil.copyfile(
                        os.path.join(srcdir, *dirnames, filename),
                        os.path.join(destdir, data_dir_name, dest_sample_name + filename_ext))
                    
                    # copy annotations
                    with open(os.path.join(srcdir, *dirnames, filename_base + ".txt"), "r") as src_annotations_file:
                        with open(os.path.join(destdir, data_dir_name, dest_sample_name + ".txt"), "w") as dest_annotations_file:
                            for line in src_annotations_file.readlines():
                                line = line.strip()
                                if line == "":
                                    continue

                                clsid, cx, cy, w, h = line.split(' ')
                                clsid = int(clsid)

                                clsid = name_to_id[src_obj_names[clsid]]
                                print("{} {} {} {} {}".format(clsid, cx, cy, w, h), file = dest_annotations_file)
                    
                    # add sample to list file
                    print("data/{}/{}{}".format(data_dir_name, dest_sample_name, filename_ext), file = dest_list_file)
    
    with open(os.path.join(destdir, "obj.names"), "w") as file:
        for name in obj_names:
            print(name, file = file)
    
    with open(os.path.join(destdir, "obj.data"), "w") as file:
        print("classes = {}".format(len(obj_names)), file = file)
        print("train = data/train.txt", file = file)
        print("names = data/obj.names", file = file)
        print("backup = backup/", file = file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str, nargs = "+", default = [])
    ap.add_argument("-dest", type = str)
    args = ap.parse_args()

    merge_datasets(args.src, args.dest)
