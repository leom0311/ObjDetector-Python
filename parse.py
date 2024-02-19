import os
import cv2
import json
import numpy as np

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
            with open(os.path.join("./annotations/_", name + '.json'), 'w') as convert_fp: 
                convert_fp.write(json.dumps(annos))

def get_color(idx, total):
    total = 127 * 255 * 255 / total * idx + 127 * 255 * 255
    G = int(total / 255 / 255)
    B = int((total - G * 255 * 255) / 255)
    R = total - G * 255 * 255 - B * 255
    return (G, B, R)

def draw_img_on_json(json_file):
    classes = []
    max_x = 0
    max_y = 0
    min_x = 9999
    min_y = 9999
    with open("./annotations/_/" + json_file, 'r') as fp:
        data = json.loads(fp.read())
        for e in data:
            cls = int(e["class"])
            x1 = float(e["location"]["x1"])
            y1 = float(e["location"]["y1"])
            x2 = float(e["location"]["x2"])
            y2 = float(e["location"]["y2"])
            if max_x < x2:
                max_x = x2
            if max_y < y2:
                max_y = y2
            if min_x > x1:
                min_x = x1
            if min_y > y1:
                min_y = y1
            try:
                classes.index(cls)
            except:
                classes.append(cls)
        print(max_x, max_y)
        image = np.zeros((int(max_y - min_y), int(max_x - min_x), 3), dtype=np.uint8)
        for e in data:
            left_top = (int(e["location"]["x1"] - int(min_x)), int(e["location"]["y1"] - int(min_y)))
            right_bottom = (int(e["location"]["x2"] - int(min_x)), int(e["location"]["y2"] - int(min_y)))
            thickness = 2
            cv2.rectangle(image, left_top, right_bottom, get_color(classes.index(int(e["class"])), len(classes)), thickness)
            
        cv2.imshow('Image', image)
        cv2.waitKey(0)
            
        

read_obj_names()
convert_type()

draw_img_on_json("00001216.json")