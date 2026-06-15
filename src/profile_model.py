import time
from contextlib import nullcontext

import torch
from fvcore.nn import FlopCountAnalysis

from net.moce_ir import MoCEIR
from options import base_parser, moce_ir, moce_ir_s
from utils.profile_utils import count_parameters, format_number


def profile_options():
    base_args = base_parser().parse_known_args()[0]

    if base_args.model == "MoCE_IR_S":
        parser = moce_ir_s(base_parser())
    elif base_args.model == "MoCE_IR":
        parser = moce_ir(base_parser())
    else:
        raise NotImplementedError(f"Model '{base_args.model}' not found.")

    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional checkpoint path to load.")
    parser.add_argument("--input_size", nargs=3, type=int, default=[3, 256, 256], metavar=("C", "H", "W"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--warmup", type=int, default=20, help="Warmup inference iterations.")
    parser.add_argument("--runs", type=int, default=100, help="Timed inference iterations.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for timing.")
    parser.add_argument("--skip_macs", action="store_true", help="Skip fvcore MAC calculation.")

    return parser.parse_args()


def build_model(opt):
    return MoCEIR(
        dim=opt.dim,
        num_blocks=opt.num_blocks,
        num_dec_blocks=opt.num_dec_blocks,
        levels=len(opt.num_blocks),
        heads=opt.heads,
        num_refinement_blocks=opt.num_refinement_blocks,
        topk=opt.topk,
        num_experts=opt.num_exp_blocks,
        rank=opt.latent_dim,
        with_complexity=opt.with_complexity,
        depth_type=opt.depth_type,
        stage_depth=opt.stage_depth,
        rank_type=opt.rank_type,
        complexity_scale=opt.complexity_scale,
    )


def load_checkpoint(model, checkpoint_path):
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("net."):
            key = key[len("net."):]
        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if missing:
        print(f"Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {len(unexpected)}")


def select_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def measure_inference_time(model, sample, warmup, runs, amp):
    device = sample.device
    model.eval()

    with torch.no_grad(), autocast_context(device, amp):
        for _ in range(warmup):
            _ = model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / runs) * 1000.0
    fps = 1000.0 / avg_ms
    return avg_ms, fps


def main():
    opt = profile_options()
    device = select_device(opt.device)

    model = build_model(opt)
    load_checkpoint(model, opt.checkpoint_path)
    model = model.to(device).eval()

    sample = torch.randn(1, *opt.input_size, device=device)
    parameter_counts = count_parameters(model)

    print("\nModel Profile")
    print(f"Model                  : {opt.model}")
    print(f"Input size             : {tuple(opt.input_size)}")
    print(f"Device                 : {device}")
    print(f"Trainable params       : {format_number(parameter_counts['trainable'])}")
    print(f"Non-trainable params   : {format_number(parameter_counts['non_trainable'])}")
    print(f"Total params           : {format_number(parameter_counts['total'])}")

    if not opt.skip_macs:
        with torch.no_grad():
            macs = FlopCountAnalysis(model, sample).total()
        print(f"MACs / fvcore FLOPs    : {macs / 1e9:.3f} G")

    avg_ms, fps = measure_inference_time(model, sample, opt.warmup, opt.runs, opt.amp)
    print(f"Inference time         : {avg_ms:.3f} ms/image")
    print(f"Throughput             : {fps:.2f} images/s")
    print(f"Warmup / timed runs    : {opt.warmup} / {opt.runs}")


if __name__ == "__main__":
    main()
