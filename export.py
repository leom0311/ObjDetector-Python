from collections import namedtuple
import torch
import torch.quantization
import torch.jit
import torch.onnx
import argparse

from .model import ObjectDetectorModel, prepare_for_qat, fuse_modules, fuse_batchnorm
from .metadata import OBJ_DETECTOR_MD_V1

ModelOutputNames = [
    "logit0", "correction0",
    "logit1", "correction1",
    "logit2", "correction2",
    "logit3", "correction3",
    "logit4", "correction4",
    "logit5", "correction5",
]

DeployModelOutput = namedtuple("DeployModelOutput", ModelOutputNames)

class DeployModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.quant              = torch.quantization.QuantStub  ()
        self.dequant_class      = torch.quantization.DeQuantStub()
        self.dequant_correction = torch.quantization.DeQuantStub()
        self.detector           = ObjectDetectorModel(OBJ_DETECTOR_MD_V1)
    
    def forward(self, image : torch.Tensor):
        image = self.quant(image)
        detector_outputs = self.detector(image)
        result = dict()

        for i, (logit_map, correction_map) in enumerate(detector_outputs):
            logit_map       = self.dequant_class(logit_map)
            correction_map  = self.dequant_correction(correction_map)

            result["logit{}".format(i)] = logit_map
            result["correction{}".format(i)] = correction_map
        
        return DeployModelOutput(**result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", type = str)
    ap.add_argument("-out", type = str)
    args = ap.parse_args()

    export_half = True

    #torch.backends.quantized.engine = "onednn"

    device = torch.device("cuda")

    print("Loading model")
    ck = torch.load(args.src, map_location = "cpu")

    model = DeployModel()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    model = fuse_batchnorm(model)
    if export_half:
        model.half()
    model.to(device)

    input = torch.rand((1, 3, 1025, 769), dtype = torch.float16 if export_half else torch.float, device = device)

    #jit_model = torch.jit.trace(model, input)
    #torch.jit.save(jit_model, args.out)

    print("Exporting ONNX")
    torch.onnx.export(
        model,
        input,
        args.out + ".onnx",
        input_names = ["image"],
        output_names = ModelOutputNames)
    
    # import to tensorrt
    print("Exporting TRT")
    import tensorrt
    
    trt_logger = tensorrt.Logger(tensorrt.Logger.INFO)

    builder = tensorrt.Builder(trt_logger)
    network = builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, trt_logger)

    config.max_workspace_size = 4 * 1024 * 1024 * 1024
    config.set_flag(tensorrt.BuilderFlag.FP16)

    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(args.out + ".onnx", "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    
    plan = builder.build_serialized_network(network, config)
    with open(args.out + ".trt", "wb") as file:
        file.write(plan)

if __name__ == "__main__":
    main()
