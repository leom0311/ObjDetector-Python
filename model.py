from typing import Callable, OrderedDict, Tuple
import torch
import torch.nn
import torch.nn.intrinsic
import torch.nn.quantized

import complexity_analysis as CA
from .metadata import ObjectDetectorMetadata

def activation_none(channels : int) -> torch.nn.Module:
    return torch.nn.Identity()

def activation_relu(channels : int) -> torch.nn.Module:
    return torch.nn.ReLU(inplace = True)

def activation_silu(channels : int) -> torch.nn.Module:
    return torch.nn.SiLU(inplace = True)

DEFAULT_ACTIVATION      = activation_silu
INTERM_FEATURE_DIM      = 128
INTERM_FEATURE_DILATION = 2
SHARE_MODULES           = False

class ConvBnAct(torch.nn.Module):
    def __init__(
            self,
            in_channels : int, out_channels : int, kernel_size : Tuple[int,int],
            activation : Callable[[int], torch.nn.Module],
            stride : Tuple[int,int] = (1,1), padding : Tuple[int,int] = (0,0),
            dilation : int = 1, groups : int = 1,
            has_bn : bool = True):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = not has_bn)
        self.bn   = torch.nn.BatchNorm2d(out_channels) if has_bn else torch.nn.Identity()
        self.act  = activation(out_channels)
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        input = self.conv(input)
        input = self.bn(input)
        input = self.act(input)
        return input

    def analyze_complexity(self, input_size):
        return CA.analyze_complexity_sequential(OrderedDict([
            ("conv", self.conv),
            ("bn"  , self.bn  ),
            ("act" , self.act ),
        ]), input_size)

    def fuse_modules(self) -> torch.nn.Module:
        if isinstance(self.bn, torch.nn.BatchNorm2d):
            if isinstance(self.act, torch.nn.ReLU):
                self.conv = torch.nn.intrinsic.ConvBnReLU2d(self.conv, self.bn, self.act)
                self.bn   = torch.nn.Identity()
                self.act  = torch.nn.Identity()
            else:
                self.conv = torch.nn.intrinsic.ConvBn2d(self.conv, self.bn)
                self.bn   = torch.nn.Identity()
        elif isinstance(self.bn, torch.nn.Identity):
            if isinstance(self.act, torch.nn.ReLU):
                self.conv = torch.nn.intrinsic.ConvReLU2d(self.conv, self.act)
                self.act  = torch.nn.Identity()

        return self
    
    def fuse_batchnorm(self):
        if isinstance(self.bn, torch.nn.BatchNorm2d):
            scale = 1 / (self.bn.running_var + self.bn.eps).sqrt()
            bias  = -self.bn.running_mean * scale

            if self.bn.weight is not None:
                scale *= self.bn.weight.data
                bias  *= self.bn.weight.data
            if self.bn.bias is not None:
                bias  += self.bn.bias.data
            
            weight = self.conv.weight.data * scale[:, None, None, None]
            if self.conv.bias is not None:
                bias += self.conv.bias.data
            
            self.conv = torch.nn.Conv2d(
                self.conv.in_channels, self.conv.out_channels,
                self.conv.kernel_size, self.conv.stride, self.conv.padding,
                dilation = self.conv.dilation, groups = self.conv.groups, bias = True)
            
            self.conv.weight.data = weight
            self.conv.bias.data   = bias

            self.bn = torch.nn.Identity()




class ResidualBlock(torch.nn.Module):
    def __init__(self, channels : int, activation : Callable[[int], torch.nn.Module], dilation : int = 1, has_bn : bool = True):
        super().__init__()

        self.a = ConvBnAct(channels, channels, (3,3), padding = (dilation, dilation), dilation = dilation, has_bn = has_bn, activation = activation)
        self.b = ConvBnAct(channels, channels, (3,3), padding = (dilation, dilation), dilation = dilation, has_bn = has_bn, activation = activation_none)

        self.f_add = torch.nn.quantized.FloatFunctional()
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        shortcut = input

        input = self.a(input)
        input = self.b(input)

        return self.f_add.add(shortcut, input)
    
    def analyze_complexity(self, input_size):
        complexity, out_size = CA.analyze_complexity_sequential(OrderedDict([
            ("a", self.a),
            ("b", self.b),
        ]), input_size)

        if input_size != out_size:
            raise Exception("Output shape of residual inner module does not match with input shape.")
        
        return complexity, input_size

def sequential_block(num : int, generator : Callable[[], torch.nn.Module]):
    layers = []
    for _ in range(0, num):
        layers.append(generator())
    return torch.nn.Sequential(*layers)


class PreprocessorModule(torch.nn.Module):
    def __init__(self, activation : Callable[[int], torch.nn.Module]):
        super().__init__()

        # x1
        self.l0 = ConvBnAct(3, 32, (3,3), stride = (2,2), padding = (1,1), activation = activation)

        # x2
        self.l1_r = sequential_block(3, lambda : ResidualBlock(32, dilation = 1, activation = activation))
        self.l1_s = ConvBnAct(32, 64, (3,3), stride = (1,1), padding = (1,1), dilation = 1, activation = activation)

        # x4 (dilation: 2)
        self.l2_r = sequential_block(4, lambda : ResidualBlock(64, dilation = 2, activation = activation))
        self.l2_s = ConvBnAct(64, 128, (3,3), stride = (1,1), padding = (2,2), dilation = 2, activation = activation)

        # # x8 (dilation: 4)
        # self.l3_r = sequential_block(4, lambda : ResidualBlock(128, dilation = 4, activation = activation))
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        input = self.l0(input)
        input = self.l1_r(input)
        input = self.l1_s(input)
        input = self.l2_r(input)
        input = self.l2_s(input)
        # input = self.l3_r(input)
        return input
    
    def analyze_complexity(self, input_size):
        return CA.analyze_complexity_sequential(OrderedDict([
            ("l0", self.l0),
            ("l1_r", self.l1_r),
            ("l1_s", self.l1_s),
            ("l2_r", self.l2_r),
            ("l2_s", self.l2_s),
        #     ("l3_r", self.l3_r),
        ]), input_size)

class OutputModule(torch.nn.Module):
    def __init__(self, metadata : ObjectDetectorMetadata, activation : Callable[[int], torch.nn.Module]):
        super().__init__()

        dilation = INTERM_FEATURE_DILATION
        padding  = (dilation, dilation)

        self.fc1 = ConvBnAct(
            INTERM_FEATURE_DIM,
            512,
            (3, 3),
            dilation    = dilation,
            padding     = padding,
            has_bn      = not SHARE_MODULES,
            activation  = activation)
        
        self.fc2 = ConvBnAct(
            512, 512,
            (1, 1),
            has_bn      = not SHARE_MODULES,
            activation  = activation)

        self.fc_logit = torch.nn.Conv2d(
            512,
            metadata.num_aspect_levels * (len(metadata.class_names) + 1),
            (1, 1))
        self.fc_correction = torch.nn.Conv2d(
            512,
            metadata.num_aspect_levels * 4,
            (1, 1))
        
        assert self.fc_logit.bias is not None
        
        self.fc_logit.bias.data.fill_(-10.)
        for i in range(0, metadata.num_aspect_levels):
            self.fc_logit.bias.data[i * (len(metadata.class_names) + 1)] = 0.
    
    def forward(self, input : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self.fc1(input)
        input = self.fc2(input)
        return self.fc_logit(input), self.fc_correction(input)
    
    def analyze_complexity(self, input_size):
        complexity, input_size = CA.analyze_complexity_sequential(OrderedDict([
            ("fc1", self.fc1),
            ("fc2", self.fc2)
        ]), input_size)

        complexity, size_logit = CA.analyze_complexity_sequential(OrderedDict([
            ("fc_logit", self.fc_logit)
        ]), input_size, complexity)
        complexity, size_correction = CA.analyze_complexity_sequential(OrderedDict([
            ("fc_correction", self.fc_correction)
        ]), input_size, complexity)

        return complexity, (size_logit, size_correction)

class FeatureDownScaleModule(torch.nn.Module):
    def __init__(self, activation : Callable[[int], torch.nn.Module]):
        super().__init__()

        dilation = INTERM_FEATURE_DILATION
        has_bn   = not SHARE_MODULES

        self.scale = ConvBnAct(
            INTERM_FEATURE_DIM, INTERM_FEATURE_DIM,
            (3,3),
            stride = (2,2), dilation = dilation, padding = (dilation, dilation),
            has_bn      = has_bn,
            activation  = activation)
        
        self.resid = sequential_block(3, lambda : ResidualBlock(
            INTERM_FEATURE_DIM, dilation = dilation, has_bn = has_bn, activation = activation))
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        input = self.scale(input)
        input = self.resid(input)
        return input
    
    def analyze_complexity(self, input_size):
        return CA.analyze_complexity_sequential(OrderedDict([
            ("scale", self.scale),
            ("resid", self.resid),
        ]), input_size)


class ObjectDetectorModel(torch.nn.Module):
    def __init__(self, metadata : ObjectDetectorMetadata, activation : Callable[[int], torch.nn.Module] = DEFAULT_ACTIVATION):
        super().__init__()

        self.num_mip_levels = metadata.num_mip_levels
        self.prep           = PreprocessorModule(activation = activation)

        if SHARE_MODULES:
            self.downscale      = FeatureDownScaleModule(activation = activation)
            self.output         = OutputModule(metadata, activation = activation)
        else:
            for i in range(1, self.num_mip_levels):
                self.__setattr__("downscale_{}".format(i), FeatureDownScaleModule(activation = activation))
            for i in range(0, self.num_mip_levels):
                self.__setattr__("output_{}".format(i), OutputModule(metadata, activation = activation))
    
    def get_downscale_module(self, i : int) -> torch.nn.Module:
        if SHARE_MODULES:
            return self.downscale
        else:
            return self.__getattr__("downscale_{}".format(i)) # type: ignore
    
    def get_output_module(self, i : int) -> torch.nn.Module:
        if SHARE_MODULES:
            return self.output
        else:
            return self.__getattr__("output_{}".format(i)) # type: ignore
    
    def forward(self, input : torch.Tensor) -> list:
        input = self.prep(input)

        outputs = []
        outputs.append(self.get_output_module(0)(input))

        for i in range(1, self.num_mip_levels):
            input = self.get_downscale_module(i)(input)
            outputs.append(self.get_output_module(i)(input))
        
        return outputs
    
    def analyze_complexity(self, input_size):
        complexity, feat_shape = CA.analyze_complexity_sequential(OrderedDict([
            ("prep", self.prep),
        ]), input_size)

        output_sizes = []

        complexity, out_size = CA.analyze_complexity_sequential(OrderedDict([
            ("output_0", self.output)
        ]), feat_shape, complexity)
        output_sizes.append(out_size)

        for i in range(1, self.num_mip_levels):
            complexity, feat_shape = CA.analyze_complexity_sequential(OrderedDict([
                ("downscale_{}".format(i), self.downscale)
            ]), feat_shape, complexity)

            complexity, out_size = CA.analyze_complexity_sequential(OrderedDict([
                ("output_{}".format(i), self.output)
            ]), feat_shape, complexity)
            output_sizes.append(out_size)
        
        return complexity, output_sizes


def fuse_modules(module : torch.nn.Module) -> torch.nn.Module:
    for name, child in module.named_children():
        child = fuse_modules(child)
        module.__setattr__(name, child)
    
    if hasattr(module, "fuse_modules"):
        module = module.fuse_modules()
    
    return module

def fuse_batchnorm(module : torch.nn.Module) -> torch.nn.Module:
    for name, child in module.named_children():
        child = fuse_batchnorm(child)
        module.__setattr__(name, child)
    
    if hasattr(module, "fuse_batchnorm"):
        module.fuse_batchnorm()
    
    return module

def prepare_for_qat(module : torch.nn.Module) -> torch.nn.Module:
    # qconfig = torch.quantization.QConfig(
    #     activation = torch.quantization.FakeQuantize.with_args(
    #         observer    = torch.quantization.MovingAverageMinMaxObserver,
    #         dtype       = torch.qint8,
    #         quant_min   = -128,
    #         quant_max   = 127),
    #     weight = torch.quantization.FakeQuantize.with_args(
    #         observer    = torch.quantization.MovingAveragePerChannelMinMaxObserver,
    #         dtype       = torch.qint8,
    #         quant_min   = -128,
    #         quant_max   = 127,
    #         qscheme     = torch.per_channel_symmetric))
    
    qconfig = torch.quantization.QConfig(
        activation = torch.quantization.FakeQuantize.with_args(
            observer    = torch.quantization.MovingAverageMinMaxObserver,
            dtype       = torch.quint8,
            quant_min   = 0,
            quant_max   = 255),
        weight = torch.quantization.FakeQuantize.with_args(
            observer    = torch.quantization.MovingAverageMinMaxObserver,
            dtype       = torch.qint8,
            quant_min   = -128,
            quant_max   = 127))
    
    module.qconfig = qconfig # type: ignore
    module = torch.quantization.prepare_qat(module, inplace = True)
    return module


if __name__ == "__main__":
    import time
    from .metadata import OBJ_DETECTOR_MD_V1

    model = ObjectDetectorModel(OBJ_DETECTOR_MD_V1)

    complexity, output_sizes = model.analyze_complexity((3, 513, 513))
    CA.print_complexity(complexity)

    for i, shape in enumerate(output_sizes):
        print("output from mip level {} : {}".format(i, shape))

    print("testing speed on cuda device...")
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    input = torch.randn((1, 3, 513, 513), dtype = torch.float, device = device)

    # warm up
    with torch.no_grad():
        model(input)
    
    # measure execution time
    num_exec = 100
    start_time = time.time()
    for i in range(0, num_exec):
        with torch.no_grad():
            model(input)
    torch.cuda.synchronize()
    end_time = time.time()

    print("execution time: {} seconds".format((end_time - start_time) / num_exec))
