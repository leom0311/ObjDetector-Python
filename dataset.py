import os
import math
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.utils.data
import torchvision.transforms as T
from PIL import Image, ImageDraw
import random
import numpy
import io

from .metadata import ObjectDetectorMetadata

HEATMAP_OVERLAP = 0.4

class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            basedir         : str,
            metadata        : ObjectDetectorMetadata,
            output_size     : Tuple[int, int]                         = (1025, 769),
            scale_range     : Tuple[float, float]                     = (1 / 2., 1 * 2.),
            image_transform : Callable[[Image.Image], torch.Tensor]   = T.ToTensor(),
            image_file_ext  : str                                     = ".PNG"):
        super().__init__()

        self.metadata           = metadata
        self.datadir            = os.path.join(basedir, "obj_train_data")
        self.output_size        = output_size
        self.image_transform    = image_transform
        self.prior_box_sizes    = metadata.calc_prior_box_sizes()

        min_scale, max_scale = scale_range
        self.min_log_scale  = math.log(min_scale)
        self.max_log_scale  = math.log(max_scale)

        label_mapping = self.read_label_mapping(os.path.join(basedir, "obj.names"))

        self.image_list = []
        for filename in os.listdir(self.datadir):
            stem, ext = os.path.splitext(filename)
            if ext != ".txt":
                continue

            image_filename = stem + image_file_ext
            if not os.path.isfile(os.path.join(self.datadir, image_filename)):
                print("  annotation '{}' does not have associated image file, ignored.".format(filename))
            
            annotations = self.read_annotation_file(os.path.join(self.datadir, filename), label_mapping)
            if len(annotations) > 0:
                self.image_list.append(( image_filename, annotations ))
    
    def read_label_mapping(self, file_path : str) -> List[int]:
        result = []
        with open(file_path, "r") as file:
            for line in file.readlines():
                name = line.strip()
                if name == "":
                    continue

                try:
                    mapped_id = self.metadata.class_names.index(name)
                except ValueError:
                    print("  class '{}' is not supported by model, ignored.".format(name))
                    mapped_id = -1
                
                result.append(mapped_id)
        
        return result

    def read_annotation_file(self, file_path : str, label_mapping : List[int]) -> List[Tuple[int, float, float, float, float]]:
        result = []
        with open(file_path, "r") as file:
            for line in file.readlines():
                line = line.strip()
                if line == "":
                    continue

                label, cx, cy, rw, rh = line.split(' ')
                mapped_label = label_mapping[int(label)]

                if mapped_label >= 0:
                    result.append((
                        mapped_label,
                        float(cx),
                        float(cy),
                        float(rw),
                        float(rh)
                    ))
        
        return result
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, index : int) -> dict:
        filename, annotations = self.image_list[index]

        image = Image.open(os.path.join(self.datadir, filename)).convert("RGB")

        original_width  = image.width
        original_height = image.height

        out_width, out_height = self.output_size

        # 1025, 769
        # max_log_scale = 0.69314718055994530941723212145818
        # min_log_scale = -0.69314718055994530941723212145818
        # out_width / original_width = 0.53385416666666666666666666666667
        # out_height / original_height = 0.71203703703703703703703703703704

        crop_scale  = math.exp(random.random() * (self.max_log_scale - self.min_log_scale) + self.min_log_scale) * min(out_width / original_width, out_height / original_height)
        crop_left   = random.random() * (original_width  - out_width  / crop_scale)
        crop_top    = random.random() * (original_height - out_height / crop_scale)

        # crop image
        image = image.transform(
            size = (out_width, out_height),
            method = Image.Transform.AFFINE,
            data = [
                1/crop_scale, 0, crop_left,
                0, 1/crop_scale, crop_top
            ],
            resample = Image.Resampling.BILINEAR)
        
        image = self.image_transform(image)

        if random.random() < 0.5:
            # crop area
            min_win_left    = max(0, int(-crop_left * crop_scale))
            max_win_right   = min(out_width, int((original_width - crop_left) * crop_scale))
            min_win_top     = max(0, int(-crop_top * crop_scale))
            max_win_bottom  = min(out_height, int((original_height - crop_top) * crop_scale))

            win_width   = random.randint(0, max_win_right - min_win_left)
            win_height  = random.randint(0, max_win_bottom - min_win_top)
            win_left    = random.randint(min_win_left, max_win_right  - win_width )
            win_top     = random.randint(min_win_top , max_win_bottom - win_height)

            win_right   = win_left + win_width
            win_bottom  = win_top  + win_height

            image[:, :, :win_left]      = 0
            image[:, :, win_right:]     = 0
            image[:, :win_top, :]       = 0
            image[:, win_bottom:, :]    = 0
        else:
            win_left    = 0
            win_top     = 0
            win_right   = out_width
            win_bottom  = out_height

        # make class & corrections map
        class_map_list      = []
        correction_map_list = []
        map_width           = (out_width  - 1) // self.metadata.first_level_scale + 1
        map_height          = (out_height - 1) // self.metadata.first_level_scale + 1

        for _ in range(0, self.metadata.num_mip_levels):
            class_map = torch.tensor([0, -1], dtype = torch.int32)[None, :, None, None]
            class_map = class_map.repeat(self.metadata.num_aspect_levels, 1, map_height, map_width)

            correction_map = torch.zeros(
                size  = (self.metadata.num_aspect_levels, 4, map_height, map_width),
                dtype = torch.float)

            class_map_list.append(class_map)
            correction_map_list.append(correction_map)

            map_width   = (map_width  - 1) // 2 + 1
            map_height  = (map_height - 1) // 2 + 1

        for label, cx, cy, width, height in annotations:
            cx      = (cx * original_width - crop_left) * crop_scale
            cy      = (cy * original_height - crop_top) * crop_scale
            width   = width  * original_width  * crop_scale
            height  = height * original_height * crop_scale

            left    = cx - width / 2
            right   = cx + width / 2
            top     = cy - height / 2
            bottom  = cy + height / 2

            if right <= win_left or left >= win_right or bottom <= win_top or top >= win_bottom:
                continue

            cx      /= self.metadata.first_level_scale
            cy      /= self.metadata.first_level_scale
            width   /= self.metadata.first_level_scale
            height  /= self.metadata.first_level_scale

            mip_level, aspect_level = self.metadata.decode_box_size(width, height)

            self.add_object(class_map_list, correction_map_list, mip_level, aspect_level, label, cx, cy, width, height)

            if max(0, min(right, win_right) - max(left, win_left)) * max(0, min(bottom, win_bottom) - max(top, win_top)) >= 0.8 * (right - left) * (bottom - top):
            #if left >= win_left and right <= win_right and top >= win_top and bottom <= win_bottom:
                self.unmark_background(class_map_list, mip_level, aspect_level, label, cx, cy)
        
        result = dict()
        result["image"] = image
        for i, m in enumerate(class_map_list):
            result["class_{}".format(i)] = m
        for i, m in enumerate(correction_map_list):
            result["correction_{}".format(i)] = m
        
        return result

    def add_object(
            self,
            class_maps : List[torch.Tensor], correction_maps : List[torch.Tensor],
            mip_level : float, aspect_level : float,
            label : int, cx : float, cy : float, width : float, height : float
            ) -> None:
        
        aspect_level    = min(max(aspect_level, 0), self.metadata.num_aspect_levels - 1)
        mip_level       = int(math.floor(mip_level))
        scale           = math.pow(2, mip_level)

        if mip_level >= 0 and mip_level < self.metadata.num_mip_levels:
            self.add_object_to_mip_level(
                class_maps[mip_level], correction_maps[mip_level],
                aspect_level,
                label, cx / scale, cy / scale, width / scale, height / scale)
        
        mip_level   += 1
        scale       *= 2

        if mip_level >= 0 and mip_level < self.metadata.num_mip_levels:
            self.add_object_to_mip_level(
                class_maps[mip_level], correction_maps[mip_level],
                aspect_level,
                label, cx / scale, cy / scale, width / scale, height / scale)

    def add_object_to_mip_level(
            self,
            class_map : torch.Tensor, correction_map : torch.Tensor, aspect_level : float,
            label : int, cx : float, cy : float, width : float, height : float
            ) -> None:
        
        aspect_level_i = int(math.floor(aspect_level))
        if aspect_level_i >= 0 and aspect_level_i < self.metadata.num_aspect_levels and abs(aspect_level_i - aspect_level) <= (1 + HEATMAP_OVERLAP) / 2:
            prior_width, prior_height = self.prior_box_sizes[aspect_level_i]

            self.add_object_to_aspect_level(
                class_map, correction_map, aspect_level_i,
                label, cx, cy, width - prior_width, height - prior_height)
        
        aspect_level_i += 1
        if aspect_level_i >= 0 and aspect_level_i < self.metadata.num_aspect_levels and abs(aspect_level_i - aspect_level) <= (1 + HEATMAP_OVERLAP) / 2:
            prior_width, prior_height = self.prior_box_sizes[aspect_level_i]

            self.add_object_to_aspect_level(
                class_map, correction_map, aspect_level_i,
                label, cx, cy, width - prior_width, height - prior_height)

    def add_object_to_aspect_level(
            self,
            class_map : torch.Tensor, correction_map : torch.Tensor, aspect_level : int,
            label : int, cx : float, cy : float, width_correction : float, height_correction : float
            ) -> None:
        
        *_, map_height, map_width = class_map.shape

        min_x = int(math.ceil(cx - (1 + HEATMAP_OVERLAP) / 2))
        min_y = int(math.ceil(cy - (1 + HEATMAP_OVERLAP) / 2))
        max_x = int(math.ceil(cx + (1 + HEATMAP_OVERLAP) / 2))
        max_y = int(math.ceil(cy + (1 + HEATMAP_OVERLAP) / 2))

        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, map_width)
        max_y = min(max_y, map_height)

        if min_x < max_x and min_y < max_y:
            y, x = torch.nonzero(class_map[aspect_level, 0, min_y : max_y, min_x : max_x] == 0, as_tuple = True)
            y += min_y
            x += min_x

            class_map[aspect_level, 1, y, x] = label + 1

            correction_map[aspect_level, 0, y, x] = (cx - x.float())
            correction_map[aspect_level, 1, y, x] = (cy - y.float())
            correction_map[aspect_level, 2, y, x] = width_correction
            correction_map[aspect_level, 3, y, x] = height_correction

    def unmark_background(
            self,
            class_maps : List[torch.Tensor],
            mip_level : float, aspect_level : float,
            label : int, cx : float, cy : float
            ) -> None:

            mip_level = round(mip_level)
            if mip_level < 0 or mip_level >= self.metadata.num_mip_levels:
                return

            aspect_level = round(aspect_level)
            if aspect_level < 0 or aspect_level >= self.metadata.num_aspect_levels:
                return
            
            scale = math.pow(2, mip_level)
            cx = round(cx / scale)
            cy = round(cy / scale)

            cls_map  = class_maps[mip_level]
            *_, map_height, map_width = cls_map.shape

            if cx >= 0 and cx < map_width and cy >= 0 and cy < map_height:
                cls_map[aspect_level, 0, cy, cx] = -1


def convert_object_detection_maps(values : Dict[str, torch.Tensor]):
    for key in values.keys():
        if key == "image":
            values[key] = values[key].float() / 255
        elif key.startswith("class_"):
            values[key] = values[key].int() - 1
        elif key.startswith("correction_"):
            values[key] = (values[key].float() - 128) / 8



class ObjectDetectionBinaryDatasetAccessor:
    def __init__(self, metadata : ObjectDetectorMetadata, image_size : Tuple[int, int]):
        super().__init__()
    
        self.metadata   = metadata
        self.image_size = image_size

        width, height = image_size
        width   = (width  + metadata.first_level_scale - 1) // metadata.first_level_scale
        height  = (height + metadata.first_level_scale - 1) // metadata.first_level_scale

        map_sizes : List[Tuple[int, int]] = list()
        for _ in range(0, metadata.num_mip_levels):
            map_sizes.append(( width, height ))
            width   = (width  + 1) // 2
            height  = (height + 1) // 2
        
        self.map_sizes = map_sizes
    
    def record_size(self) -> int:
        size = 0

        width, height = self.image_size
        size += 3 * width * height

        for width, height in self.map_sizes:
            size += 2 * self.metadata.num_aspect_levels * width * height # class map
            size += 4 * self.metadata.num_aspect_levels * width * height # correction map
        
        return size
    
    def write_record(self, file : io.BufferedWriter, values : Dict[str, torch.Tensor]):
        # write image
        t = values["image"]
        width, height = self.image_size

        if t.shape != torch.Size([3, height, width]):
            raise Exception("Incorrect tensor shape.")
        
        t = (t * 255 + 0.5).floor().clamp(0, 255).byte()
        t = numpy.ascontiguousarray(t.numpy())
        file.write(t)

        # write class & correction maps
        for i, (width, height) in enumerate(self.map_sizes):
            # write class map
            t = values["class_{}".format(i)]

            if t.shape != torch.Size([self.metadata.num_aspect_levels, 2, height, width]):
                raise Exception("Incorrect tensor shape.")
            
            t = (t + 1).byte()
            t = numpy.ascontiguousarray(t.numpy())
            file.write(t)

            # write correction map
            t = values["correction_{}".format(i)]

            if t.shape != torch.Size([self.metadata.num_aspect_levels, 4, height, width]):
                raise Exception("Incorrect tensor shape.")
            
            t = (t * 8 + 128.5).floor().byte()
            t = numpy.ascontiguousarray(t.numpy())
            file.write(t)
    
    def read_record(self, file : io.BufferedReader, convert : bool = True) -> Dict[str, torch.Tensor]:
        result : Dict[str, torch.Tensor] = dict()

        # read image
        width, height = self.image_size
        t = numpy.fromfile(file, dtype = numpy.uint8, count = 3 * height * width)
        t = torch.from_numpy(t).reshape(3, height, width)
        result["image"] = t

        # read class & correction maps
        for i, (width, height) in enumerate(self.map_sizes):
            # read class map
            t = numpy.fromfile(file, dtype = numpy.uint8, count = self.metadata.num_aspect_levels * 2 * height * width)
            t = torch.from_numpy(t).reshape(self.metadata.num_aspect_levels, 2, height, width)
            result["class_{}".format(i)] = t

            # read correction map
            t = numpy.fromfile(file, dtype = numpy.uint8, count = self.metadata.num_aspect_levels * 4 * height * width)
            t = torch.from_numpy(t).reshape(self.metadata.num_aspect_levels, 4, height, width)
            result["correction_{}".format(i)] = t
        
        if convert:
            convert_object_detection_maps(result)
        
        return result


class ObjectDetectorBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, metadata : ObjectDetectorMetadata, image_size : Tuple[int, int], filepath : str, convert : bool):
        super().__init__()

        self.accessor       = ObjectDetectionBinaryDatasetAccessor(metadata, image_size)
        self.record_size    = self.accessor.record_size()
        self.convert        = convert

        with open(filepath, "rb") as file:
            file.seek(0, os.SEEK_END)
            self.count = file.tell() // self.record_size
        
        self.filepath = filepath
        self.file     = None
    
    def __len__(self) -> int:
        return self.count
    
    def __getitem__(self, n : int) -> Dict[str, torch.Tensor]:
        if n < 0 or n >= self.count:
            raise IndexError()
        
        if self.file is None:
            self.file = open(self.filepath, "rb")
        
        self.file.seek(n * self.record_size, os.SEEK_SET)
        return self.accessor.read_record(self.file, convert = self.convert)


def debug_dump_maps(
        metadata    : ObjectDetectorMetadata,
        prior_sizes : List[Tuple[float, float]],
        cls_map     : torch.Tensor,
        corr_map    : torch.Tensor,
        mip_level   : int,
        draw        : Optional[ImageDraw.ImageDraw]):
    
    l_aspect_level, l_cy, l_cx = torch.nonzero(cls_map[:, 1, :, :] > 0, as_tuple = True)
    scale = math.pow(2, mip_level) * metadata.first_level_scale

    for aspect_level, cy, cx in zip(l_aspect_level.tolist(), l_cy.tolist(), l_cx.tolist()):
        label : int = int(cls_map[aspect_level, 1, cy, cx].item()) - 1
        d_cx, d_cy, d_w, d_h = corr_map[aspect_level, :, cy, cx].tolist()

        is_definite = True

        w, h = prior_sizes[aspect_level]
        name = metadata.class_names[label]
        if cls_map[aspect_level, 0, cy, cx] >= 0:
            name += "?"
            is_definite = False

        x1 = round( ((cx + d_cx) - (w + d_w) / 2) * scale )
        x2 = round( ((cx + d_cx) + (w + d_w) / 2) * scale )
        y1 = round( ((cy + d_cy) - (h + d_h) / 2) * scale )
        y2 = round( ((cy + d_cy) + (h + d_h) / 2) * scale )

        print("{}\t{}, {}, {}, {}".format(name, x1, y1, x2, y2))

        if draw is not None and is_definite:
            draw.rectangle((x1, y1, x2, y2), outline = "red")

def test_dataset_loader(dataset_dir : str):
    import matplotlib.pyplot as P
    import torchvision.transforms.functional as TF

    from .metadata import OBJ_DETECTOR_MD_V1

    prior_sizes = OBJ_DETECTOR_MD_V1.calc_prior_box_sizes()
    ds = ObjectDetectionDataset(dataset_dir, OBJ_DETECTOR_MD_V1)
    while True:
        v = ds[random.randint(0, len(ds) - 1)]

        image = v["image"]
        image = TF.to_pil_image(image)
        draw = ImageDraw.Draw(image)

        for i in range(0, OBJ_DETECTOR_MD_V1.num_mip_levels):
            debug_dump_maps(OBJ_DETECTOR_MD_V1, prior_sizes, v["class_{}".format(i)], v["correction_{}".format(i)], i, draw)

        P.imshow(image)
        P.show()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str)
    args = ap.parse_args()

    test_dataset_loader(args.src)
