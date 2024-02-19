import os
import cv2
import json

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
        if ext.upper() == ".PNG" or ext.upper() == ".JSON":
            continue
        width = 0
        height = 0
        if ext.upper() == ".TXT":
            image = cv2.imread(os.path.join("./annotations/_/", name + ".PNG"))
            height, width, _ = image.shape
        
        with open(os.path.join("./annotations/_/", file), "r") as fp:
            annos = []
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

                x1 = round((c_x - _w / 2), 3)
                y1 = round((c_y - _h / 2), 3)
                x2 = round((c_x + _w / 2), 3)
                y2 = round((c_y + _h / 2), 3)

                item = {
                    "class": cls,
                    "location": {
                        "x1":x1,
                        "y1":y1,
                        "x2":x2,
                        "y2":y2,
                    }
                }
                annos.append(item)
            with open("./annotations/_/" + name + '.json', 'w') as convert_fp: 
                convert_fp.write(json.dumps(annos))
                
read_obj_names()
convert_type()