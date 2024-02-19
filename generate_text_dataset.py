import argparse
import os
from typing import Dict, List, Set, Tuple
from PIL import Image, ImageDraw, ImageFont
import random
import math
import matplotlib.pyplot as P
import torchvision.transforms.functional as TF


EXCLUDE_FONT_LIST = set([x.lower() for x in [
    "BSSYM7",
    "FRSCRIPT",
    "holomdl2",
    "ITCBLKAD",
    "ITCEDSCR",
    "KUNSTLER",
    "marlett",
    "MTEXTRA",
    "OUTLOOK",
    "PALSCRI",
    "PARCHM",
    "REFSPCL",
    "segmdl2",
    "SegoeIcons",
    "symbol",
    "webdings",
    "wingding",
    "WINGDNG2",
    "WINGDNG3",
]])

def exp_uniform(min : float, max : float) -> float:
    log_min = math.log(min)
    log_max = math.log(max)
    return math.exp(random.random() * (log_max - log_min) + log_min)

def scan_bg_images(dirpath : str, filelist : List[str], extensions : Set[str]):
    for child_name in os.listdir(dirpath):
        child_path = os.path.join(dirpath, child_name)
        if os.path.isdir(child_path):
            scan_bg_images(child_path, filelist, extensions)
        else:
            _, ext = os.path.splitext(child_name)
            if ext.lower() in extensions:
                filelist.append(child_path)

def scan_font_files(dir : str) -> List[str]:
    result = list()
    for child_name in os.listdir(dir):
        child_path = os.path.join(dir, child_name)
        if not os.path.isfile(child_path):
            continue

        stem, ext = os.path.splitext(child_name)
        if ext.lower() == ".ttf" and stem.lower() not in EXCLUDE_FONT_LIST:
            result.append(child_path)
    
    return result

def generate_string() -> str:
    if random.random() < 0.5:
        return random.choice(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    else:
        num_digits = random.randint(1, 5)
        num_decimals = random.randint(0, 3)

        result = ""
        for _ in range(num_digits):
            result += str(random.randint(0, 9))
        if num_decimals > 0:
            result += "."
            for _ in range(num_decimals):
                result += str(random.randint(0, 9))
        
        return result

def create_random_font(font_list : List[str]) -> ImageFont.FreeTypeFont:
    font_path = random.choice(font_list)
    # print(font_path)
    return ImageFont.truetype(font_path, size = int(exp_uniform(10, 50)))

def draw_text_to_image(
        draw            : ImageDraw.ImageDraw,
        image_width     : int,
        image_height    : int,
        text            : str,
        font            : ImageFont.FreeTypeFont,
        class_mapping   : Dict[str, int],
        objects         : List[Tuple[int, float, float, float, float]]
    ):

    bbox_left, bbox_top, bbox_right, bbox_bottom = draw.textbbox((0,0), text, font = font)
    text_width  = bbox_right - bbox_left
    text_height = bbox_bottom - bbox_top

    origin_x = random.random() * (image_width - text_width)
    origin_y = random.random() * (image_height - text_height)

    draw.text(
        (origin_x - bbox_left, origin_y - bbox_top),
        text,
        font = font,
        fill = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(190, 255)))
    
    last_x = 0
    for i in range(0, len(text)):
        new_x = draw.textlength(text[0:i+1], font = font)

        char_cx     = origin_x + (new_x + last_x) / 2
        char_cy     = origin_y + text_height / 2
        char_width  = new_x - last_x
        char_height = text_height

        objects.append((
            class_mapping[text[i : i+1]],
            char_cx / image_width,
            char_cy / image_height,
            char_width / image_width,
            char_height / image_height
        ))

        last_x = new_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-font-dir", type = str, default = r"C:\Windows\Fonts")
    ap.add_argument("-bg-images", type = str)
    ap.add_argument("-out", type = str, default = None)
    ap.add_argument("-count", type = int, default = 100)
    args = ap.parse_args()

    print("Scanning background images...")
    bg_image_files = list()
    scan_bg_images(args.bg_images, bg_image_files, set([".bmp", ".png", ".jpg", ".jpeg"]))

    print("Scanning font files...")
    font_list = scan_font_files(args.font_dir)

    class_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    class_mapping = dict()
    for i, ch in enumerate(class_names):
        class_mapping[ch] = i

    if args.out is not None:
        os.makedirs(os.path.join(args.out, "obj_train_data"), exist_ok = True)

    sample_name_list = list()

    for sample_id in range(0, args.count):
        print(sample_id)
        
        image = Image.open(random.choice(bg_image_files))
        image = TF.resize(image, (1024, 1024))
        image = image.convert("RGBA")
        
        overlay = Image.new(mode = "RGBA", size = (image.width, image.height), color = (0,0,0,0))
        draw = ImageDraw.Draw(overlay)

        objects = list()

        for _ in range(random.randint(50, 100)):
            draw_text_to_image(draw, image.width, image.height, generate_string(), create_random_font(font_list), class_mapping, objects)

        if args.out is not None:
            sample_name = "{}".format(sample_id)

            overlaid = Image.alpha_composite(image, overlay).convert("RGB")
            overlaid.save(os.path.join(args.out, "obj_train_data", sample_name + ".png"))

            with open(os.path.join(args.out, "obj_train_data", sample_name + ".txt"), "w") as file:
                for id, cx, cy, w, h in objects:
                    print("{} {} {} {} {}".format(id, cx, cy, w, h), file = file)
            
            sample_name_list.append(sample_name)

        if False:
            for _, cx, cy, w, h in objects:
                draw.rectangle((
                    (cx-w/2) * image.width,
                    (cy-h/2) * image.height,
                    (cx+w/2) * image.width,
                    (cy+h/2) * image.height,
                ), outline = "red")
            
            overlaid = Image.alpha_composite(image, overlay).convert("RGB")
            P.imshow(overlaid)
            P.show()

    if args.out is not None:
        with open(os.path.join(args.out, "train.txt"), "w") as list_file:
            for sample_name in sample_name_list:
                print("data/obj_train_data/{}.png".format(sample_name), file = list_file)
        
        with open(os.path.join(args.out, "obj.names"), "w") as file:
            for name in class_mapping:
                print(name, file = file)
        
        with open(os.path.join(args.out, "obj.data"), "w") as file:
            print("classes = {}".format(len(class_names)), file = file)
            print("train = data/train.txt", file = file)
            print("names = data/obj.names", file = file)
            print("backup = backup/", file = file)


if __name__ == "__main__":
    main()
