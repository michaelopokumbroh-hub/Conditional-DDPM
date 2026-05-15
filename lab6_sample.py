import os, json
import torch
from torchvision.utils import make_grid, save_image

from lab6_step1_dataset import labels_to_onehot
from lab6_step2_model   import UNet, LinearNoiseScheduler, MODEL_CONFIG


# Load model

def load_trained_model(
    ckpt_path = "./checkpoints/ddpm_epoch0100.pth",
    device    = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(ckpt_path):
        save_dir = "./checkpoints"
        available = sorted([
            f for f in os.listdir(save_dir)
            if f.startswith("ddpm_epoch") and f.endswith(".pth")
        ], reverse=True)
        if available:
            ckpt_path = os.path.join(save_dir, available[0])
            print(f"  [Auto] Using best available: {ckpt_path}")
        else:
            raise FileNotFoundError(f"No checkpoints found in {save_dir}/")

    print(f"Loading: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = UNet(MODEL_CONFIG).to(device)
    model.load_state_dict(ckpt["model"])

    if "ema" in ckpt and ckpt["ema"]:
        print("  Applying EMA weights...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in ckpt["ema"]:
                    param.data.copy_(ckpt["ema"][name])
        print("  EMA weights applied ✓")
    else:
        print("  No EMA found — using raw model weights")

    model.eval()
    print(f"  Epoch: {ckpt.get('epoch', '?')}")

    scheduler = LinearNoiseScheduler(
        num_timesteps=1000, beta_start=1e-4, beta_end=0.02
    )
    return model, scheduler



# Reverse diffusion sampling

@torch.no_grad()
def sample_images(model, scheduler, condition,
                  img_size=64, device=None,
                  record_every=None):
    if device is None:
        device = next(model.parameters()).device

    B  = condition.shape[0]
    T  = scheduler.num_timesteps
    xt = torch.randn(B, 3, img_size, img_size, device=device)

    frames = []
    for t in reversed(range(T)):
        t_batch    = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model(xt, t_batch, condition)
        xt, x0     = scheduler.sample_prev_timestep(xt, noise_pred, t)

        
        if record_every is not None:
            if t % record_every == 0 or t == T-1 or t == 0:
                frames.append(xt.clone().cpu())

    return xt.cpu(), frames



# Generate images for test.json or new_test.json

@torch.no_grad()
def generate_images(
    model,
    scheduler,
    json_path     = "./test.json",
    obj_json_path = "./objects.json",
    save_folder   = "./images/test",
    img_size      = 64,
):
    device = next(model.parameters()).device
    model.eval()
    os.makedirs(save_folder, exist_ok=True)

    with open(json_path) as f:
        conditions = json.load(f)
    with open(obj_json_path) as f:
        obj2idx = json.load(f)

    n = len(conditions)
    print(f"\nGenerating {n} images  [{os.path.basename(json_path)}]")

    all_labels = torch.stack([
        labels_to_onehot(c, obj2idx) for c in conditions
    ])

    all_images, _ = sample_images(
        model, scheduler,
        condition=all_labels.to(device),
        img_size=img_size,
        device=device,
    )

    imgs_01 = ((all_images + 1) / 2).clamp(0, 1)
    for i in range(n):
        save_image(imgs_01[i], os.path.join(save_folder, f"{i}.png"))

    grid = make_grid(imgs_01, nrow=8, padding=2)
    save_image(grid, os.path.join(save_folder, "grid.png"))

    print(f"  Saved {n} PNGs  -> {save_folder}/")
    print(f"  Saved grid     -> {save_folder}/grid.png")

    return all_images, all_labels



# Evaluate with evaluator

def run_evaluation(images, labels):
    import sys
    sys.path.insert(0, ".")
    from evaluator import evaluation_model

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = evaluation_model()
    acc       = evaluator.eval(images.to(device), labels.to(device))
    return acc



# Denoising process grid

@torch.no_grad()
def generate_denoising_process(
    model,
    scheduler,
    obj_json_path = "./objects.json",
    save_path     = "./images/denoising_process.png",
    n_frames      = 10,
    img_size      = 64,
):
    device        = next(model.parameters()).device
    target_labels = ["red sphere", "cyan cylinder", "cyan cube"]
    print(f"\nDenoising process: {target_labels}")

    with open(obj_json_path) as f:
        obj2idx = json.load(f)

    cond = labels_to_onehot(
        target_labels, obj2idx
    ).unsqueeze(0).to(device)

    T            = scheduler.num_timesteps
    record_every = T // n_frames

    _, frames = sample_images(
        model, scheduler,
        condition    = cond,
        img_size     = img_size,
        device       = device,
        record_every = record_every,
    )

    
    frames_01 = [
        ((f + 1) / 2).clamp(0, 1).squeeze(0)
        for f in frames
    ][:n_frames]

    print(f"  Frames: {len(frames_01)}  (left=noisy  right=clean)")

    grid = make_grid(
        torch.stack(frames_01),
        nrow=len(frames_01), padding=2
    )
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        exist_ok=True
    )
    save_image(grid, save_path)
    print(f"  Saved -> {save_path}")
    return grid



# run_all()

def run_all(model, scheduler):
    print("=" * 55)
    print("  Lab 6  |  Inference + Evaluation")
    print("=" * 55)

    imgs_test, lbl_test = generate_images(
        model, scheduler,
        json_path   = "./test.json",
        save_folder = "./images/test",
    )
    acc_test = run_evaluation(imgs_test, lbl_test)
    print(f"  >>> test.json accuracy     : {acc_test:.4f}")

    imgs_new, lbl_new = generate_images(
        model, scheduler,
        json_path   = "./new_test.json",
        save_folder = "./images/new_test",
    )
    acc_new = run_evaluation(imgs_new, lbl_new)
    print(f"  >>> new_test.json accuracy : {acc_new:.4f}")

    generate_denoising_process(model, scheduler)

    print("\n" + "=" * 55)
    print("  FINAL RESULTS")
    print("=" * 55)
    print(f"  test.json     : {acc_test:.4f}")
    print(f"  new_test.json : {acc_new:.4f}")
    print("=" * 55)
    print("\nFiles saved:")
    print("  images/test/0.png ... 31.png")
    print("  images/test/grid.png")
    print("  images/new_test/0.png ... 31.png")
    print("  images/new_test/grid.png")
    print("  images/denoising_process.png")

    return acc_test, acc_new


if __name__ == "__main__":
    model, scheduler = load_trained_model()
    run_all(model, scheduler)
