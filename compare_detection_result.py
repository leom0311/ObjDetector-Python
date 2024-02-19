import os
import argparse
from typing import List, Optional, Tuple
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import math

from .metadata import OBJ_DETECTOR_MD_V1
from .model import ObjectDetectorModel, fuse_batchnorm, prepare_for_qat
from .result_retrieval import ObjectDetectionResultRetriever


FP16 = True


class ImageProcessor:
    def __init__(self, device : torch.device, model_file_path : str, threshold : float, model_input_size : Tuple[int, int] = (1025, 769)):
        super().__init__()

        self.device = device
        self.metadata = OBJ_DETECTOR_MD_V1
        self.model_input_width, self.model_input_height = model_input_size
        self.threshold = threshold

        # load model
        sd = torch.load(model_file_path, map_location = "cpu")["state_dict"]
        filtered_sd = dict()
        prefix = "detector."
        for key in sd.keys():
            if key.startswith(prefix):
                filtered_sd[key[len(prefix) :]] = sd[key]
        
        self.model = ObjectDetectorModel(self.metadata)
        self.model.load_state_dict(filtered_sd)
        self.model = fuse_batchnorm(self.model)
        self.model.to(device)
        if FP16:
            self.model.half()
        self.model.eval()

        # create result retriever
        self.result_retriever = ObjectDetectionResultRetriever(self.metadata, device)
    
    def process(self, image : Image.Image) -> List[Tuple[str, float, float, float, float, float]]:
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
        if FP16:
            model_input = model_input.half()

        with torch.no_grad():
            maps = self.model(model_input)
            class_id, class_prob, coordinates = self.result_retriever(maps, 0, threshold = self.threshold)

        class_id    = class_id.cpu()
        class_prob  = class_prob.cpu()
        coordinates = coordinates.cpu()

        result = []
        for id, prob, (cx, cy, w, h) in zip(class_id.tolist(), class_prob.tolist(), coordinates.tolist()):
            cx = (cx - x0) / (original_width  * scale)
            cy = (cy - y0) / (original_height * scale)
            w  = w / (original_width  * scale)
            h  = h / (original_height * scale)

            result.append(( self.metadata.class_names[id], cx, cy, w, h, 1 / (1 + math.exp(-prob)) ))
        return result

def read_class_names(srcdir : str) -> List[str]:
    result = []

    with open(os.path.join(srcdir, "obj.names"), "r") as file:
        for line in file.readlines():
            line = line.strip()
            if line != "":
                result.append(line)
    
    return result

def read_annotations(srcpath : str, class_names : List[str]) -> List[Tuple[str, float, float, float, float]]:
    result = []

    with open(srcpath, "r") as file:
        for line in file.readlines():
            line = line.strip()
            if line == "":
                continue

            id, cx, cy, w, h = line.split(' ')
            id = int(id)
            cx = float(cx)
            cy = float(cy)
            w  = float(w)
            h  = float(h)

            result.append(( class_names[id], cx, cy, w, h ))
    
    return result

def intersection_ratio(loc1 : Tuple[float, float, float, float], loc2 : Tuple[float, float, float, float]) -> float:
    x1, y1, w1, h1 = loc1
    x2, y2, w2, h2 = loc2

    rx = min(min(w1, w2), max(0, (w1+w2)/2 - abs(x1 - x2))) / max(w1, w2)
    ry = min(min(h1, h2), max(0, (h1+h2)/2 - abs(y1 - y2))) / max(h1, h2)

    return rx * ry

class CompareResult:
    expected    : Optional[Tuple[str, float, float, float, float]]
    actual      : Optional[Tuple[str, float, float, float, float, float]]

def compare_annotations(
        ground_truth     : List[Tuple[str, float, float, float, float]],
        actual_result    : List[Tuple[str, float, float, float, float, float]],
        min_intersection : float = 0.8,
        min_confidence   : float = 0.5
    ) -> List[CompareResult]:
    result = []

    for actual_name, actual_x, actual_y, actual_w, actual_h, confidence in actual_result:
        best_match_idx      = -1
        best_match_score    = 0

        for i, (gt_name, gt_x, gt_y, gt_w, gt_h) in enumerate(ground_truth):
            score = intersection_ratio((actual_x, actual_y, actual_w, actual_h), (gt_x, gt_y, gt_w, gt_h))
            if score > best_match_score:
                best_match_idx   = i
                best_match_score = score
        
        if best_match_idx < 0:
            # spurious detection
            cr = CompareResult()
            cr.expected = None
            cr.actual   = (actual_name, actual_x, actual_y, actual_w, actual_h, confidence)
            result.append(cr)
        else:
            gt_name, gt_x, gt_y, gt_w, gt_h = ground_truth[best_match_idx]
            if best_match_score >= min_intersection:
                if actual_name != gt_name:
                    # wrong detection
                    cr = CompareResult()
                    cr.expected = (gt_name, gt_x, gt_y, gt_w, gt_h)
                    cr.actual   = (actual_name, actual_x, actual_y, actual_w, actual_h, confidence)
                    result.append(cr)
                
                ground_truth.pop(best_match_idx)
            elif actual_name == gt_name:
                # wrong detection
                cr = CompareResult()
                cr.expected = (gt_name, gt_x, gt_y, gt_w, gt_h)
                cr.actual   = (actual_name, actual_x, actual_y, actual_w, actual_h, confidence)
                result.append(cr)
                
                ground_truth.pop(best_match_idx)
            elif confidence < min_confidence:
                # too low confidence
                cr = CompareResult()
                cr.expected = (gt_name, gt_x, gt_y, gt_w, gt_h)
                cr.actual   = (actual_name, actual_x, actual_y, actual_w, actual_h, confidence)
                result.append(cr)
    
    for missed in ground_truth:
        cr = CompareResult()
        cr.expected = missed
        cr.actual   = None
        result.append(cr)

    return result

def verify_model_detection_results(dataset_dir : str, model_path : str, threshold : float, upper_threshold : float):
    dataset_class_names = read_class_names(dataset_dir)
    proc = ImageProcessor(torch.device("cuda"), model_path, threshold = threshold)

    with open(os.path.join(dataset_dir, "train.txt"), "r") as list_file:
        for line in list_file.readlines():
            line = line.strip()
            if line == "":
                continue

            dir1, *dirnames, filename = line.split('/')
            if dir1 != "data":
                raise Exception("Expected first component of relative path to be 'data'")
            
            stem, _ = os.path.splitext(filename)
            
            image = Image.open(os.path.join(dataset_dir, *dirnames, filename))
            actual_detections = proc.process(image)
            expected_detections = read_annotations(os.path.join(dataset_dir, *dirnames, stem + ".txt"), dataset_class_names)

            compare_res = compare_annotations(expected_detections, actual_detections, min_confidence = upper_threshold)

            if len(compare_res) > 0:
                print(filename)
                for cr in compare_res:
                    if cr.expected is None:
                        assert cr.actual is not None
                        name, x, y, w, h, confidence = cr.actual

                        print("    Spurious : {} {} {} {} '{}' P={:.4f}".format(
                            round((x-w/2) * image.width),
                            round((y-h/2) * image.height),
                            round(w * image.width),
                            round(h * image.height),
                            name,
                            confidence
                        ))
                    elif cr.actual is None:
                        name, x, y, w, h = cr.expected

                        print("    Missed   : {} {} {} {} '{}'".format(
                            round((x-w/2) * image.width),
                            round((y-h/2) * image.height),
                            round(w * image.width),
                            round(h * image.height),
                            name
                        ))
                    else:
                        n1, x1, y1, w1, h1, confidence = cr.actual
                        n2, x2, y2, w2, h2 = cr.expected

                        if n1 != n2:
                            print("    Wrong    : {} {} {} {} '{}' <-- '{}' P={:.4f}".format(
                                round((x2-w2/2) * image.width),
                                round((y2-h2/2) * image.height),
                                round(w2 * image.width),
                                round(h2 * image.height),
                                n1,
                                n2,
                                confidence
                            ))
                        else:
                            print("    Loc/Conf : '{}' {} {} {} {} <-- {} {} {} {} P={:.4f}".format(
                                n1,
                                round((x1-w1/2) * image.width),
                                round((y1-h1/2) * image.height),
                                round(w1 * image.width),
                                round(h1 * image.height),
                                round((x2-w2/2) * image.width),
                                round((y2-h2/2) * image.height),
                                round(w2 * image.width),
                                round(h2 * image.height),
                                confidence
                            ))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-dataset", type = str)
    ap.add_argument("-checkpoint", type = str)
    ap.add_argument("-threshold", type = float, default = 0.5)
    ap.add_argument("-upper-threshold", type = float, default = 0.7)
    args = ap.parse_args()

    verify_model_detection_results(args.dataset, args.checkpoint, args.threshold, args.upper_threshold)
