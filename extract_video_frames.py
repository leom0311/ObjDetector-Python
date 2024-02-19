import argparse
import os
import torchvision.transforms.functional as TF

from .video_reader import VideoReader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str)
    ap.add_argument("-out", type = str)
    ap.add_argument("-region", type = int, nargs = 4, default = None)
    ap.add_argument("-interval", type = int, default = 1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok = True)
    video_reader = VideoReader()
    video_reader.open(args.src)

    if args.region is not None:
        left, top, width, height = args.region

    frame_no = 0
    frame = video_reader.get_frame()
    while frame is not None:
        print(frame_no)

        if frame_no % args.interval == 0:
            if args.region is not None:
                frame = TF.crop(frame, top, left, height, width)

            frame.save(os.path.join(args.out, "{:08d}.png".format(frame_no)))
        
        frame_no += 1

        frame = video_reader.get_frame()


if __name__ == "__main__":
    main()
