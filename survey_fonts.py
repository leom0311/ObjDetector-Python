import argparse
import os
from typing import List
from PIL import Image, ImageDraw, ImageFont

def scan_font_files(dir : str) -> List[str]:
    result = list()
    for child_name in os.listdir(dir):
        child_path = os.path.join(dir, child_name)
        if not os.path.isfile(child_path):
            continue

        _, ext = os.path.splitext(child_name)
        if ext.lower() == ".ttf":
            result.append(child_path)
    
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-font-dir", type = str, default = r"C:\Windows\Fonts")
    ap.add_argument("-out-dir", type = str)
    args = ap.parse_args()

    font_files = scan_font_files(args.font_dir)
    os.makedirs(args.out_dir, exist_ok = True)

    text = "0123456789"
    for font_path in font_files:
        font = ImageFont.truetype(font_path, size = 20)
        left, top, right, bottom = font.getbbox(text)

        image = Image.new(mode = "RGB", size = (right - left, bottom - top), color = "white")
        draw = ImageDraw.Draw(image)
        draw.text((-left, -top), text, font = font, fill = "black")

        *_, filename = os.path.split(font_path)
        stem, _ = os.path.splitext(filename)
        image.save(os.path.join(args.out_dir, stem + ".png"))

if __name__ == "__main__":
    main()
