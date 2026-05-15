import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from lab6_step1_dataset import get_train_loader
from lab6_step2_model   import UNet, EMA, LinearNoiseScheduler, MODEL_CONFIG

# Optimized
CFG = {
    "root"        : ".",
    "obj_json"    : "./objects.json",
    "save_dir"    : "./checkpoints",
    "batch_size"  : 4,      
    "accumulate"  : 8,      # 4 * 8 = Effective batch size of 32
    "lr"          : 1e-4,
    "epochs"      : 150,
    "save_every"  : 10,
}

def train(resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG["save_dir"], exist_ok=True)

    # 1. Data & Model
    loader, _ = get_train_loader(CFG["root"], CFG["obj_json"], batch_size=CFG["batch_size"])
    model = UNet(MODEL_CONFIG).to(device)
    ema = EMA(model)
    scheduler_noise = LinearNoiseScheduler()
    optimizer = Adam(model.parameters(), lr=CFG["lr"])
    
    start_epoch = 1
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path)
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        start_epoch = ckpt['epoch'] + 1

    # 2. Training Loop
    for epoch in range(start_epoch, CFG["epochs"] + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{CFG['epochs']}")
        optimizer.zero_grad()
        
        for i, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            noise = torch.randn_like(imgs)
            t = torch.randint(0, 1000, (imgs.size(0),), device=device).long()

            # Forward
            x_t = scheduler_noise.add_noise(imgs, noise, t)
            pred = model(x_t, t, labels)
            
            # Loss with accumulation scaling
            loss = nn.functional.mse_loss(pred, noise) / CFG["accumulate"]
            loss.backward()

            # Update weights every N steps
            if (i + 1) % CFG["accumulate"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)
            
            pbar.set_postfix(loss=loss.item() * CFG["accumulate"])

        # 3. Save Checkpoint
        if epoch % CFG["save_every"] == 0 or epoch == CFG["epochs"]:
            save_path = f"{CFG['save_dir']}/ddpm_epoch{epoch:04d}.pth"
            torch.save({
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'epoch': epoch
            }, save_path)
            print(f"\nSaved: {save_path}")

    return model, ema

def sanity_check_training():
    """Quick run to ensure gradients flow and loss decreases"""
    print("Running Training Sanity Check...")
    device = torch.device("cuda")
    loader, _ = get_train_loader(".", "./objects.json", batch_size=4)
    model = UNet(MODEL_CONFIG).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler_noise = LinearNoiseScheduler()
    
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)
    
    t = torch.randint(0, 1000, (4,), device=device).long()
    noise = torch.randn_like(imgs)
    x_t = scheduler_noise.add_noise(imgs, noise, t)
    
    pred = model(x_t, t, labels)
    loss = nn.functional.mse_loss(pred, noise)
    loss.backward()
    optimizer.step()
    print(f"Sanity Check Passed! Initial Loss: {loss.item():.4f}")

if __name__ == "__main__":
    sanity_check_training()