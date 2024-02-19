import os
import argparse
from typing import List, Optional, Tuple
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import math

from .metadata import ObjectDetectorMetadata, OBJ_DETECTOR_MD_V1
from .model import ObjectDetectorModel, fuse_modules, prepare_for_qat
from .result_retrieval import ObjectDetectionResultRetriever

class TestModel(torch.nn.Module):
    def __init__(self, metadata : ObjectDetectorMetadata, quantized : bool):
        super().__init__()

        self.detector           = ObjectDetectorModel(metadata)
        self.quant              = torch.quantization.QuantStub  () if quantized else torch.nn.Identity()
        self.dequant_class      = torch.quantization.DeQuantStub() if quantized else torch.nn.Identity()
        self.dequant_correction = torch.quantization.DeQuantStub() if quantized else torch.nn.Identity()
    
    def forward(self, input : torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        input = self.quant(input)
        maps  = self.detector(input)
        maps  = [(self.dequant_class(cls_map), self.dequant_correction(corr_map)) for cls_map, corr_map in maps]
        return maps

class ImageProcessor:
    def __init__(self, device : torch.device, model_file_path : str, quantized : bool = False, model_input_size : Tuple[int, int] = (961, 541)):
        super().__init__()

        self.device = device
        self.metadata = OBJ_DETECTOR_MD_V1
        self.model_input_width, self.model_input_height = model_input_size

        # load model
        sd = torch.load(model_file_path, map_location = "cpu")["state_dict"]
        
        self.model = TestModel(self.metadata, quantized = quantized)
        if quantized:
            self.model = fuse_modules(self.model)
            self.model = prepare_for_qat(self.model)
            
        self.model.load_state_dict(sd)
        self.model.to(device)
        self.model.eval()

        # create result retriever
        self.result_retriever = ObjectDetectionResultRetriever(self.metadata, device)
    
    def process(self, image : Image.Image) -> List[Tuple[int, float, float, float, float]]:
        original_width  = image.width
        original_height = image.height

        scale   = min(self.model_input_width / original_width, self.model_input_height / original_height)
        x0      = (self.model_input_width  - original_width  * scale) / 2
        y0      = (self.model_input_height - original_height * scale) / 2

        image = image.transform(
            (self.model_input_width, self.model_input_height),
            method = Image.Transform.AFFINE,
            data = [
                1/scale, 0, -x0/scale,
                0, 1/scale, -y0/scale],
            resample = Image.Resampling.BILINEAR)

        model_input = TF.to_tensor(image)[None].to(self.device)

        with torch.no_grad():
            maps = self.model(model_input)
            class_id, class_prob, coordinates = self.result_retriever(maps, 0)

        class_id    = class_id.cpu()
        class_prob  = class_prob.cpu()
        coordinates = coordinates.cpu()

        result = []
        for id, prob, (cx, cy, w, h) in zip(class_id.tolist(), class_prob.tolist(), coordinates.tolist()):
            cx = (cx - x0) / (original_width  * scale)
            cy = (cy - y0) / (original_height * scale)
            w  = w / (original_width  * scale)
            h  = h / (original_height * scale)

            result.append((id, cx, cy, w, h))
        return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-model", type = str)
    ap.add_argument("-src", type = str)
    ap.add_argument("-out", type = str, default = None)
    ap.add_argument("-quantized", default = False, action = "store_true")
    ap.add_argument("-extensions", type = str, nargs = "+", default = [".bmp", ".png", ".jpg", ".jpeg", ".BMP", ".PNG", ".JPG", ".JPEG"])
    ap.add_argument("-show-result", default = False, action = "store_true")
    args = ap.parse_args()

    from PIL import ImageDraw
    import matplotlib.pyplot as P

    src_path = args.src
    out_dir = args.out

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok = True)
        os.makedirs(os.path.join(out_dir, "obj_train_data"), exist_ok = True)

    print("Initializing...")
    processor = ImageProcessor(
        torch.device("cuda"),
        args.model,
        quantized = args.quantized)
    
    print("Processing...")
    if out_dir is not None:
        list_file = open(os.path.join(out_dir, "train.txt"), "w")
    else:
        list_file = None
    
    def process_file(image_file_path : str, attrib_file_name : Optional[str]):
        print("  " + image_file_path)

        image = Image.open(image_file_path).convert("RGB")
        objects = processor.process(image)

        if out_dir is not None and attrib_file_name is not None:
            with open(os.path.join(out_dir, "obj_train_data", attrib_file_name), "w") as file:
                for id, cx, cy, w, h in objects:
                    print("{} {} {} {} {}".format(id, cx, cy, w, h), file = file)
        
        if args.show_result:
            draw = ImageDraw.Draw(image)
            for id, cx, cy, w, h in objects:
                draw.rectangle(
                    (
                        int(math.floor((cx - w / 2) * image.width  + 0.5)),
                        int(math.floor((cy - h / 2) * image.height + 0.5)),
                        int(math.floor((cx + w / 2) * image.width  + 0.5)),
                        int(math.floor((cy + h / 2) * image.height + 0.5)),
                    ),
                    outline = "red")
            P.imshow(image)
            P.show()
        
        if list_file is not None and attrib_file_name is not None:
            print("data/obj_train_data/{}".format(attrib_file_name), file = list_file)

    if os.path.isdir(src_path):
        for filename in os.listdir(src_path):
            stem, ext = os.path.splitext(filename)
            if ext not in args.extensions:
                continue

            process_file(os.path.join(src_path, filename), stem + ".txt")
    else:
        process_file(src_path, None)
    
    if list_file is not None:
        list_file.close()
        del list_file
    
    if out_dir is not None:
        with open(os.path.join(out_dir, "obj.names"), "w") as file:
            for name in processor.metadata.class_names:
                print(name, file = file)
        
        with open(os.path.join(out_dir, "obj.data"), "w") as file:
            print("classes = {}".format(len(processor.metadata.class_names)), file = file)
            print("train = data/train.txt", file = file)
            print("names = data/obj.names", file = file)
            print("backup = backup/", file = file)



if __name__ == "__main__":
    main()
