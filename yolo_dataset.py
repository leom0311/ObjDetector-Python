from typing import List

class AnnotatedObject:
    class_id    : int
    cx          : float
    cy          : float
    width       : float
    height      : float

def read_object_list(filepath : str) -> List[AnnotatedObject]:
    res = list()
    
    with open(filepath, "r") as file:
        for line in file.readlines():
            line = line.strip()
            if line == "":
                continue

            label, cx, cy, w, h = line.split(' ')
            obj = AnnotatedObject()
            obj.class_id    = int(label)
            obj.cx          = float(cx)
            obj.cy          = float(cy)
            obj.width       = float(w)
            obj.height      = float(h)

            res.append(obj)
    return res

def write_object_list(filepath : str, objects : List[AnnotatedObject]):
    with open(filepath, "w") as file:
        for obj in objects:
            print("{} {} {} {} {}".format(obj.class_id, obj.cx, obj.cy, obj.width, obj.height), file = file)

def read_class_list(filepath : str) -> List[str]:
    res = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            line = line.strip()
            if line != "":
                res.append(line)
    return res
