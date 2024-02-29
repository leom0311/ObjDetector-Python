import os

dirs = [
    "C:\\Users\\Administrator\\Pictures\\training\\dataset\\fine\\v1\\"
]

result = dict()
classes = []
with open(dirs[0] + "obj.names", "r") as fp:
    for line in fp.readlines():
        line = line.strip()
        if line == "":
            continue
        result[line] = 0
        classes.append(line)
    fp.close()
for dir in dirs:
    with open(dir + "train.txt", "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == "":
                continue
            file, _ = os.path.splitext(line.split("/")[2])
            anno = dir + "obj_train_data\\" + file + ".txt"
            print(anno)
            with open(anno, "r") as an:
                for item in an.readlines():
                    item = item.strip()
                    if item == "":
                        continue
                    idx = int(item.split(" ")[0])
                    result[classes[idx]] = result[classes[idx]] + 1
                an.close()
        fp.close()
print(result)