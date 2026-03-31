import os


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "yes"

import argparse
import random
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios_ode import HeliosPipeline
from helios.scheduler.scheduling_helios import HeliosScheduler
from helios.utils.utils_base import encode_prompt, load_extra_components
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusers.models import AutoencoderKLWan


def setup_distributed_env():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def check_file_exists(args):
    basename, idx, line, output_folder = args
    uttid = f"{basename}_{idx:05d}"
    output_path = os.path.join(output_folder, f"{uttid}.pt")
    if os.path.exists(output_path):
        return None, None
    return line.strip(), uttid


def prepare_dataset_on_rank0(txt_file, output_folder, rank):
    while True:
        try:
            if rank == 0:
                basename = Path(txt_file).stem
                output_dir = Path(output_folder)

                existing_files = set()
                if output_dir.exists():
                    existing_files = {f.name for f in output_dir.iterdir() if f.is_file()}

                prompts = []
                uttids = []

                with open(txt_file, "r") as f:
                    for idx, line in enumerate(f):
                        if not line.strip():
                            continue

                        uttid = f"{basename}_{idx:05d}"
                        filename = f"{uttid}.pt"

                        if filename not in existing_files:
                            prompts.append(line.strip())
                            uttids.append(uttid)

                data_to_broadcast = [prompts, uttids]
            else:
                data_to_broadcast = [None, None]

            dist.broadcast_object_list(data_to_broadcast, src=0)
            break
        except Exception:
            continue

    return data_to_broadcast[0], data_to_broadcast[1]


def prepare_i2v_dataset_on_rank0(pt_data_dirs, output_folder, rank, max_samples=0):
    """Scan .pt data dirs for (prompt, first_frames_image) pairs, skip already-processed."""
    while True:
        try:
            if rank == 0:
                output_dir = Path(output_folder)
                existing_files = set()
                if output_dir.exists():
                    existing_files = {f.name for f in output_dir.iterdir() if f.is_file()}

                pt_paths = []
                uttids = []
                for data_dir in pt_data_dirs:
                    for f in sorted(os.listdir(data_dir)):
                        if not f.endswith(".pt"):
                            continue
                        uttid = f[:-3]
                        if f"{uttid}.pt" in existing_files:
                            continue
                        pt_paths.append(os.path.join(data_dir, f))
                        uttids.append(uttid)

                if max_samples > 0 and len(pt_paths) > max_samples:
                    random.seed(42)
                    indices = random.sample(range(len(pt_paths)), max_samples)
                    indices.sort()
                    pt_paths = [pt_paths[i] for i in indices]
                    uttids = [uttids[i] for i in indices]

                print(f"I2V ODE: {len(pt_paths)} samples to process "
                      f"(skipped {len(existing_files)} existing)")
                data_to_broadcast = [pt_paths, uttids]
            else:
                data_to_broadcast = [None, None]

            dist.broadcast_object_list(data_to_broadcast, src=0)
            break
        except Exception:
            continue

    return data_to_broadcast[0], data_to_broadcast[1]


class PromptDataset(Dataset):
    def __init__(self, prompts, uttids):
        self.prompts = prompts
        self.uttids = uttids

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "uttid": self.uttids[idx]}


class PromptImageDataset(Dataset):
    """Loads (prompt, first_frames_image) from pre-encoded .pt files for I2V ODE generation."""

    def __init__(self, pt_paths, uttids):
        self.pt_paths = pt_paths
        self.uttids = uttids

    def __len__(self):
        return len(self.pt_paths)

    def __getitem__(self, idx):
        data = torch.load(self.pt_paths[idx], map_location="cpu", weights_only=False)
        image = data["first_frames_image"]  # (3, H, W) uint8
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return {
            "prompt": data["prompt_raw"],
            "image": image,
            "uttid": self.uttids[idx],
        }


def main():
    args = parse_args()

    # =============== Environment ===============
    batch_size = 1
    dataloader_num_workers = 8
    feature_folders = args.prompt_txt_files or []
    output_folders = args.output_dirs

    if args.weight_dtype == "fp32":
        args.weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        args.weight_dtype = torch.float16
    else:
        args.weight_dtype = torch.bfloat16

    setup_distributed_env()

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()

    accelerator = Accelerator()

    # =============== Prepare Model ===============
    _subfolder = args.subfolder if args.subfolder is not None else "transformer"
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder=_subfolder,
        torch_dtype=args.weight_dtype,
        use_default_loader=args.use_default_loader,
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()
    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=torch.float32)
    if args.is_enable_stage2:
        scheduler = HeliosScheduler(
            shift=args.stage2_timestep_shift,
            stages=args.stage2_num_stages,
            stage_range=args.stage2_stage_range,
            gamma=args.stage2_scheduler_gamma,
        )
        pipe = HeliosPipeline.from_pretrained(
            args.base_model_path,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=args.weight_dtype,
        )
    else:
        pipe = HeliosPipeline.from_pretrained(
            args.base_model_path, transformer=transformer, vae=vae, torch_dtype=args.weight_dtype
        )
    pipe = pipe.to(device)

    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[1.0])

        if args.partial_path is not None:
            if not hasattr(args, "training_config"):
                from argparse import Namespace

                args.training_config = Namespace()
            args.training_config.is_enable_stage1 = True
            args.training_config.restrict_self_attn = True
            args.training_config.is_amplify_history = True
            args.training_config.is_use_gan = True
            load_extra_components(args, transformer, args.partial_path)

    if args.vae_decode_type == "once":
        pipe.vae.enable_tiling()

    transformer.eval()
    transformer.requires_grad_(False)
    vae.eval()
    vae.requires_grad_(False)

    transformer.to(device)
    vae.to(device)
    pipe.to(device)

    is_i2v = args.sample_type == "i2v"

    if is_i2v:
        # =============== I2V Mode: load (prompt, image) from .pt data dirs ===============
        if not args.pt_data_dirs:
            raise ValueError("--pt_data_dirs is required for I2V ODE generation")
        from PIL import Image as PILImage

        for output_folder in output_folders:
            os.makedirs(output_folder, exist_ok=True)

        pt_paths, uttids = prepare_i2v_dataset_on_rank0(
            args.pt_data_dirs, output_folders[0], rank, max_samples=args.i2v_max_samples
        )
        dataset = PromptImageDataset(pt_paths, uttids)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_num_workers,
            prefetch_factor=2 if dataloader_num_workers > 0 else None,
            pin_memory=True,
            drop_last=False,
        )
        dataloader = accelerator.prepare(dataloader)
        print(f"I2V Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")
        print(f"Process index: {accelerator.process_index}, World size: {accelerator.num_processes}")

        i2v_folders = [(output_folders[0], dataloader)]
    else:
        # =============== T2V Mode: load prompts from .txt files (original behavior) ===============
        i2v_folders = []
        for feature_folder, output_folder in zip(feature_folders, output_folders):
            print(f"Process {feature_folder} !")
            os.makedirs(output_folder, exist_ok=True)
            prompts, uttids = prepare_dataset_on_rank0(feature_folder, output_folder, rank)
            dataset = PromptDataset(prompts, uttids)
            dl = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=dataloader_num_workers,
                prefetch_factor=2 if dataloader_num_workers > 0 else None,
                pin_memory=True,
                drop_last=False,
            )
            dl = accelerator.prepare(dl)
            print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dl)}")
            print(f"Process index: {accelerator.process_index}, World size: {accelerator.num_processes}")
            i2v_folders.append((output_folder, dl))

    for output_folder, dataloader in i2v_folders:
        if len(dataloader) == 0:
            continue

        # =============== Main Loop ===============
        if rank == 0:
            pbar = tqdm(total=len(dataloader), desc=f"Processing ({'I2V' if is_i2v else 'T2V'})")

        for i, batch in enumerate(dataloader):
            assert len(batch["uttid"]) == 1
            uttid = batch["uttid"][0]
            prompt_raw = batch["prompt"][0]

            output_path = os.path.join(output_folder, f"{uttid}.pt")
            if os.path.exists(output_path):
                if rank == 0:
                    print(f"Skipping existing file: {output_path}")
                    pbar.update(1)
                continue

            # Prepare I2V image (PIL) if in I2V mode
            pil_image = None
            if is_i2v:
                img_tensor = batch["image"][0]  # (3, H, W) float [0, 1]
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
                pil_image = PILImage.fromarray(img_np)
                if pil_image.size != (args.width, args.height):
                    pil_image = pil_image.resize((args.width, args.height))

            with torch.no_grad():
                prompt_embed, _ = encode_prompt(
                    tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder,
                    prompt=prompt_raw,
                    device=device,
                )

                all_sections_ode = pipe(
                    prompt=prompt_raw,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=50,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(args.seed),
                    output_type="latent",
                    vae_decode_type=args.vae_decode_type,
                    # stage 1
                    history_sizes=[16, 2, 1],
                    latent_window_size=args.latent_window_size,
                    is_keep_x0=True,
                    use_dynamic_shifting=args.use_dynamic_shifting,
                    time_shift_type=args.time_shift_type,
                    # stage 2
                    is_enable_stage2=args.is_enable_stage2,
                    stage2_num_stages=args.stage2_num_stages,
                    stage2_num_inference_steps_list=args.stage2_num_inference_steps_list,
                    scheduler_type="unipc",
                    # cfg zero
                    use_cfg_zero_star=args.use_cfg_zero_star,
                    use_zero_init=args.use_zero_init,
                    zero_steps=args.zero_steps,
                    # I2V
                    image=pil_image,
                    image_noise_sigma_min=args.image_noise_sigma_min,
                    image_noise_sigma_max=args.image_noise_sigma_max,
                )

            # (Pdb) len(all_sections_ode)
            # 264 -> % 8 == 0
            # 231 -> % 7 == 0
            # 198 -> % 6 == 0
            # 165 -> % 5 == 0
            # (Pdb) len(all_sections_ode[0])
            # 3
            # (Pdb) all_sections_ode[0][0].keys()
            # dict_keys(['latents', 'timesteps', 'noise_pred'])
            # (Pdb) all_sections_ode[0][0]["timesteps"].shape
            # torch.Size([20]
            # (Pdb) all_sections_ode[0][0]["latents"].shape
            # torch.Size([20, 1, 16, 9, 12, 20])
            # (Pdb) all_sections_ode[0][0]["noise_pred"].shape
            # torch.Size([20, 1, 16, 9, 12, 20])

            processed_sections_ode = []
            for idx, section in enumerate(all_sections_ode):
                processed_section = []
                for iidx, item in enumerate(section):
                    if idx == 0:
                        if iidx == 0:
                            selected_target_timesteps = [998.5342, 902.2183, 833.9636, 783.0660]
                        elif iidx == 1:
                            selected_target_timesteps = [742.8216, 640.0038, 547.1926, 462.9951]
                        elif iidx == 2:
                            selected_target_timesteps = [385.4137, 328.6249, 253.9905, 151.5308]
                    else:
                        if iidx == 0:
                            selected_target_timesteps = [998.5342, 833.9636]
                        elif iidx == 1:
                            selected_target_timesteps = [742.8216, 547.1926]
                        elif iidx == 2:
                            selected_target_timesteps = [385.4137, 253.9905]

                    indices = []
                    actual_timesteps = item["timesteps"]
                    for target_t in selected_target_timesteps:
                        diffs = torch.abs(actual_timesteps - target_t)
                        closest_idx = torch.argmin(diffs).item()
                        indices.append(closest_idx)
                    latents_indices = indices + [-1]

                    rocessed_item = {
                        "latents": item["latents"][latents_indices],
                        "timesteps": item["timesteps"][indices],
                    }

                    processed_section.append(rocessed_item)
                processed_sections_ode.append(processed_section)
            all_sections_ode = processed_sections_ode

            temp_to_save = {
                "latent_window_size": args.latent_window_size,
                "prompt_raw": prompt_raw,
                "prompt_embed": prompt_embed,
                "ode_latents": all_sections_ode,
                "sample_type": args.sample_type,
            }
            torch.save(temp_to_save, output_path)
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({"uttid": uttid})


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video with model")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Base")
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="BestWishYsh/Helios-Mid",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--partial_path",
        type=str,
        default=None,
    )
    parser.add_argument("--use_default_loader", action="store_true")
    parser.add_argument(
        "--subfolder",
        type=str,
        default="transformer",
        help='Subfolder inside transformer_path. Use "" for flat merged dirs.',
    )

    # === Generation parameters ===
    # environment
    parser.add_argument(
        "--sample_type",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "v2v"],
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")
    # base
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=165)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--use_dynamic_shifting", action="store_true")
    parser.add_argument(
        "--time_shift_type",
        type=str,
        default="linear",
        choices=["exponential", "linear"],
    )
    parser.add_argument("--vae_decode_type", type=str, default="default", choices=["default", "once", "default_fast"])
    # stage 1
    parser.add_argument("--latent_window_size", type=int, default=9)
    # stage 2
    parser.add_argument("--is_enable_stage2", action="store_true")
    parser.add_argument("--stage2_timestep_shift", type=float, default=1.0)
    parser.add_argument("--stage2_scheduler_gamma", type=float, default=1 / 3)
    parser.add_argument("--stage2_stage_range", type=int, nargs="+", default=[0, 1 / 3, 2 / 3, 1])
    parser.add_argument("--stage2_num_stages", type=int, default=3)
    parser.add_argument("--stage2_num_inference_steps_list", type=int, nargs="+", default=[20, 20, 20])
    # cfg zero
    parser.add_argument("--use_cfg_zero_star", action="store_true")
    parser.add_argument("--use_zero_init", action="store_true")
    parser.add_argument("--zero_steps", type=int, default=1)

    # === I2V parameters ===
    parser.add_argument("--image_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--image_noise_sigma_max", type=float, default=0.135)
    parser.add_argument(
        "--pt_data_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Directories of pre-encoded .pt files (containing first_frames_image + prompt_raw). "
        "Required when --sample_type=i2v. Each .pt becomes one I2V ODE sample.",
    )
    parser.add_argument(
        "--i2v_max_samples",
        type=int,
        default=0,
        help="Max number of I2V samples to process (0 = all). Useful to cap ODE data size.",
    )

    # === Data I/O ===
    parser.add_argument(
        "--prompt_txt_files",
        type=str,
        nargs="*",
        default=None,
        help="[T2V mode] One or more .txt files, each with one prompt per line.",
    )
    parser.add_argument(
        "--output_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Output dir(s). For T2V: one per prompt_txt_file. For I2V: single dir.",
    )

    # === Prompts ===
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )

    args = parser.parse_args()

    if args.sample_type == "t2v":
        if not args.prompt_txt_files:
            parser.error("--prompt_txt_files is required for T2V mode")
        if len(args.prompt_txt_files) != len(args.output_dirs):
            parser.error("--prompt_txt_files and --output_dirs must have the same count in T2V mode")
    elif args.sample_type == "i2v":
        if not args.pt_data_dirs:
            parser.error("--pt_data_dirs is required for I2V mode")

    return args


if __name__ == "__main__":
    # from diffusers import AutoencoderKLWan
    # from diffusers.video_processor import VideoProcessor
    # from diffusers.utils import export_to_video

    # device = "cuda"
    # pretrained_model_name_or_path = "BestWishYsh/Helios-Base"
    # vae = AutoencoderKLWan.from_pretrained(
    #     pretrained_model_name_or_path,
    #     subfolder="vae",
    #     torch_dtype=torch.float32,
    # ).to(device)
    # vae.eval()
    # vae.requires_grad_(False)

    # vae_scale_factor_spatial = vae.spatial_compression_ratio
    # video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    # latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    # latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)

    # x1 = torch.load("/mnt/hdfs/data/ysh_new/userful_things_wan/ode_pairs/vidprom_filtered_extended/vidprom_filtered_extended_00011.pt", map_location="cpu")
    # vae_latents = x1["ode_latents"][-1][-1]["latents"][-1] / latents_std + latents_mean
    # vae_latents = vae_latents.to(device=device, dtype=vae.dtype)
    # video = vae.decode(vae_latents, return_dict=False)[0]
    # video = video_processor.postprocess_video(video, output_type="pil")
    # export_to_video(video[0], "output_wan.mp4", fps=30)

    main()
