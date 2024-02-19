import os
import cv2

obj_names = []
def read_obj_names() :
    with open("./annotations/obj.names", "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == "":
                continue
            obj_names.append(line)

def convert_type():
    for file in os.listdir("./annotations/_"):
        name, ext = os.path.splitext(file)
        if ext.upper() == ".PNG":
            continue

        width = 0
        height = 0
        if ext.upper() == ".TXT":
            image = cv2.imread(os.path.join("./annotations/_/", name + ".PNG"))
            height, width, _ = image.shape
            
        with open(os.path.join("./annotations/_/", file), "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                if line == "":
                    continue
                cls, c_x, c_y, _w, _h = line.split(" ")
                cls = int(cls)
                c_x = float(c_x) * width
                c_y = float(c_y) * height
                _w = float(_w) * width
                _h = float(_h) * height
                print(cls, c_x, c_y, _w, _h)


read_obj_names()
convert_type()