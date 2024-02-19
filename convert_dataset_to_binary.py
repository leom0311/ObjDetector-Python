import os
import argparse
import torch
import torch.utils.data
import torchvision.transforms as T

from .dataset import ObjectDetectionDataset, ObjectDetectionBinaryDatasetAccessor
from .metadata import OBJ_DETECTOR_MD_V1


def main(args):
    image_size = (513, 513)

    image_transform = T.Compose([
        #T.RandomInvert(),
        T.RandomApply([
            T.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5), hue = (-0.3, 0.3)),
        ]),
        T.ToTensor()
    ])
    
    dataset = [ObjectDetectionDataset(dir, OBJ_DETECTOR_MD_V1, image_size) for dir in args.dataset]
    if args.dataset_repeat > 1:
        dataset = dataset * args.dataset_repeat
    dataset = torch.utils.data.ConcatDataset(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size          = None,
        shuffle             = False,
        num_workers         = args.workers,
        pin_memory          = False,
        drop_last           = False)
    
    accessor = ObjectDetectionBinaryDatasetAccessor(OBJ_DETECTOR_MD_V1, image_size)
    
    with open(args.out, "wb") as out_file:
        for i, v in enumerate(loader):
            accessor.write_record(out_file, v)

            print("\r\033[K  {}/{}".format(i, len(loader)), end = "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"             , type = str, nargs = "+", default = [])
    ap.add_argument("--dataset-repeat"      , type = int, default = 1)
    ap.add_argument("--out"                 , type = str)
    ap.add_argument("--workers"             , type = int, default = 12)
    args = ap.parse_args()
    main(args)
