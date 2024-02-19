import os
from typing import Callable
import torch
import torch.utils.data
import torch.cuda.amp
import torch.backends.cuda
import torch.backends.cudnn
import torch.quantization
import torch.nn.intrinsic.qat
import argparse
import torchvision.transforms as T
import time
import math

from .dataset import ObjectDetectionDataset, ObjectDetectorBinaryDataset, convert_object_detection_maps

from .metadata import OBJ_DETECTOR_MD_V1
from .model import ObjectDetectorModel, fuse_modules, fuse_batchnorm, prepare_for_qat
from .loss import LossHead
from stat_util import StatTracker, AverageMeter


class BlendedActivation(torch.nn.Module):
    def __init__(self, act1_class : Callable[[], torch.nn.Module], act2_class : Callable[[], torch.nn.Module]):
        super().__init__()

        self.act1       = act1_class()
        self.act2       = act2_class()
        self.blending   = 0.
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        o1 = self.act1(input)
        o2 = self.act2(input)
        return o1 * (1 - self.blending) + o2 * self.blending

class IncreaseActivationBlend:
    def __init__(self, amount : float):
        super().__init__()
        self.amount = amount
    
    def __call__(self, module : torch.nn.Module):
        if isinstance(module, BlendedActivation):
            module.blending = min(1., module.blending + self.amount)


class TrainModel(torch.nn.Module):
    def __init__(self, qat : bool = False, **kwargs):
        super().__init__()

        self.quant              = torch.quantization.QuantStub  () if qat else torch.nn.Identity()
        self.dequant_class      = torch.quantization.DeQuantStub() if qat else torch.nn.Identity()
        self.dequant_correction = torch.quantization.DeQuantStub() if qat else torch.nn.Identity()
        self.detector           = ObjectDetectorModel(OBJ_DETECTOR_MD_V1, **kwargs)
        self.loss_head          = LossHead()
    
    def forward(self, image : torch.Tensor, **maps : dict):
        image = self.quant(image)
        detector_outputs = self.detector(image)

        loss = 0
        stats = dict()

        for i, (logit_map, correction_map) in enumerate(detector_outputs):
            loss += self.loss_head(
                self.dequant_class(logit_map),
                self.dequant_correction(correction_map),
                maps["class_{}".format(i)],
                maps["correction_{}".format(i)],
                stats,
                "mip{}".format(i)
            ) # * math.pow(4, i)
        
        return loss, stats


def main(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    compute_device = torch.device("cuda")

    use_bin_dataset = False

    if use_bin_dataset:
        train_dataset = [ObjectDetectorBinaryDataset(OBJ_DETECTOR_MD_V1, args.image_size, filepath, convert = False) for filepath in args.train_dataset]
    else:
        train_dataset = [ObjectDetectionDataset(filepath, OBJ_DETECTOR_MD_V1, args.image_size) for filepath in args.train_dataset]
        

    if args.dataset_repeat > 1:
        train_dataset = train_dataset * args.dataset_repeat

    train_dataset = torch.utils.data.ConcatDataset(train_dataset)

    extra_model_args = {}
    if args.migrate_act is not None:
        if args.migrate_act == "silu_relu":
            extra_model_args["activation"] = lambda channels : BlendedActivation(torch.nn.SiLU, torch.nn.ReLU)
        elif args.migrate_act == "relu_silu":
            extra_model_args["activation"] = lambda channels : BlendedActivation(torch.nn.ReLU, torch.nn.SiLU)
        else:
            raise Exception("Unrecognized --migrate-act option: {}".format(args.migrate_act))

    model = TrainModel(qat = (args.qat is not None), **extra_model_args)
    print(model)

    model.to(compute_device)

    epoch = 0
    
    if args.bn_merged:
        model = fuse_batchnorm(model)

    if args.qat is not None and not args.qat_migrate:
        model = fuse_modules(model)
        model = prepare_for_qat(model)

    if args.resume is not None:
        ck = torch.load(args.resume, map_location = compute_device)
        sd = ck["state_dict"]

        keys_to_delete = []
        for key in sd.keys():
            if any([(x in key) for x in args.ignore]):
                print("  ignoring: {}".format(key))
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del sd[key]

        model.load_state_dict(sd, strict = args.strict)
        epoch = ck["epoch"]
        del sd
        del ck
        del keys_to_delete
    
    if args.bn_merge_migrate:
        model = fuse_batchnorm(model)

    if args.qat is not None and args.qat_migrate:
        model = fuse_modules(model)
        model = prepare_for_qat(model)

    if args.qat is not None:
        if args.qat == "full":
            pass
        elif args.qat == "bn_frozen":
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        elif args.qat == "frozen":
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            model.apply(torch.quantization.disable_observer)
        else:
            raise Exception("Unrecognized QAT mode: {}".format(args.qat))

    #
    params = list()
    for module in model.modules():
        lr_scale = 1.
        wd_scale = 1.

        if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            wd_scale = 0.
        
        params.append({
            "params"        : module.parameters(recurse = False),
            "initial_lr"    : args.lr * lr_scale,
            "weight_decay"  : args.wd * wd_scale
        })

    if args.optimizer == "adam":
        optim = torch.optim.AdamW(params, lr = args.lr, weight_decay = args.wd)
    elif args.optimizer == "sgd":
        optim = torch.optim.SGD(params, lr = args.lr, weight_decay = args.wd)
    else:
        raise Exception("Unknown optimizer type: {}".format(args.optimizer))

    if args.lr_annealing_period is not None:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            args.lr_annealing_period,
            verbose     = False,
            last_epoch  = epoch-1)
    else:
        sched = None

    best_epoch = None
    best_loss  = None

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size          = args.batch_size,
        shuffle             = True,
        num_workers         = args.workers,
        pin_memory          = True,
        drop_last           = True,
        persistent_workers  = (args.workers > 0))

    if args.half:
        grad_scaler = torch.cuda.amp.GradScaler()
    
    save_suffix = "q" if args.qat is not None else ""

    while True:
        print("===== epoch {} =====".format(epoch))

        # train
        model.train()

        num_backtrack_lines : int = 0
        stat_tracker = StatTracker()
        mean_loss = 0.
        loss_count = 0

        last_time               = time.time()
        average_iteration_time  = AverageMeter()
        average_dataload_time   = AverageMeter()

        for i, data in enumerate(loader):
            for key in data:
                x = data[key]
                if torch.is_tensor(x):
                    data[key] = x.to(compute_device)
            
            if use_bin_dataset:
                convert_object_detection_maps(data)

            now = time.time()
            average_dataload_time.add_sample(now - last_time)
            last_time = now

            optim.zero_grad()
            if args.half:
                with torch.cuda.amp.autocast(dtype = torch.float16):
                    loss, stats = model(**data)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optim)
                grad_scaler.update()
            else:
                loss, stats = model(**data)

                loss.backward()
                optim.step()

            stat_tracker.update(stats)

            now = time.time()
            average_iteration_time.add_sample(now - last_time)
            last_time = now

            loss_count += 1
            mean_loss += (loss.item() - mean_loss) / loss_count

            if (i+1) % args.print_freq == 0 or (i+1) == len(loader):
                print("\033[A\r" * num_backtrack_lines, end = "")

                print("\r\033[K[{:5d}/{:5d}] {:8.4f} ({:8.4f}) Time {:.3f}/{:.3f}".format(
                    i+1, len(loader), loss.item(), mean_loss,
                    average_iteration_time.average, average_dataload_time.average))
                num_backtrack_lines = 1 + stat_tracker.print()

                if args.half:
                    print("\r\033[K  grad-scale : {}".format(grad_scaler.get_scale()))
                    num_backtrack_lines += 1
            
            # activation migration
            if args.migrate_act is not None:
                model.apply(IncreaseActivationBlend(args.migrate_act_speed))

        # update epoch
        if sched is not None:
            sched.step()
        epoch += 1

        #
        if args.save_dir is not None:
            torch.save({
                "state_dict"        : model.state_dict(),
                "epoch"             : epoch,
            }, os.path.join(args.save_dir, "{}{}.pth".format(epoch, save_suffix)))

# python -m obj_detection.train 
# --train-dataset C:\Users\Administrator\Pictures\train-data\v2 
# --dataset-repeat 10 
# --save-dir C:\Users\Administrator\Pictures\save\5 
# --optimizer adam 
# --lr 1e-4 
# --wd 0 
# --half 
# --batch-size 4 
# --resume C:\Users\Administrator\Pictures\save\v1\5.pth

if __name__ == "__main__":
   
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dataset"       , type = str, nargs = "+", default = [])
    ap.add_argument("--dataset-repeat"      , type = int,  default = 1)
    ap.add_argument("--image-size"          , type = int, nargs = 2, default = [513, 513])
    ap.add_argument("--save-dir"            , type = str)
    ap.add_argument("--workers"             , type = int, default = 8)
    ap.add_argument("--batch-size"          , type = int, default = 4)
    ap.add_argument("--lr"                  , type = float, default = 1e-4)
    ap.add_argument("--wd"                  , type = float, default = 1e-4)
    ap.add_argument("--lr-annealing-period" , type = int, default = None)
    ap.add_argument("--print-freq"          , type = int, default = 1)
    ap.add_argument("--resume"              , type = str, default = None)
    ap.add_argument("--optimizer"           , type = str, default = "adam")
    ap.add_argument("--half"                , default = False, action = "store_true")
    ap.add_argument("--tf32"                , default = False, action = "store_true")
    ap.add_argument("--strict"              , default = True, action = "store_true")
    ap.add_argument("--no-strict"           , dest = "strict", action = "store_false")
    ap.add_argument("--ignore"              , type = str, nargs = "+", default = [])
    ap.add_argument("--bn-merged"           , default = False, action = "store_true")
    ap.add_argument("--bn-merge-migrate"    , default = False, action = "store_true")
    ap.add_argument("--qat"                 , type = str, default = None)
    ap.add_argument("--qat-migrate"         , default = False, action = "store_true")
    ap.add_argument("--migrate-act"         , type = str, default = None)
    ap.add_argument("--migrate-act-speed"   , type = float, default = 1e-4)
    args = ap.parse_args()
    main(args)
