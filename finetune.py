import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from tqdm import tqdm

from final_train import MedSAMPINN

# ==========================================
# CONFIGURATION
# ==========================================
CHECKPOINT_PATH = "medsam_pinn_busi_epoch_50.pth"
TRAIN_DIR = "./train"
SAVE_PATH = "medsam_pinn_nerve_finetuned.pth"
EPOCHS = 25
BATCH_SIZE = 1  
LR = 5e-6 

# ==========================================
# NERVE-SPECIFIC DATASET
# ==========================================
class KaggleNerveTrainDataset(Dataset):
    def __init__(self, data_dir, image_size=1024):
        self.data_dir = data_dir
        self.image_size = image_size
        
        all_files = sorted(os.listdir(data_dir))
        raw_image_files = [f for f in all_files if f.endswith('.tif') and '_mask' not in f]
        
        self.image_files = []
        for img_f in raw_image_files:
            mask_f = img_f.replace('.tif', '_mask.tif')
            mask_path = os.path.join(data_dir, mask_f)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.max() > 0:
                    self.image_files.append(img_f)
                
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.tif', '_mask.tif')
        
        image = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, img_name)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        mask = cv2.imread(os.path.join(self.data_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask_bin = binary_fill_holes(mask > 127).astype(np.uint8)
        
        # SAFETY CATCH: Prevent crash if mask vanishes during resize/threshold
        y_indices, x_indices = np.where(mask_bin > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            bbox = np.array([0, 0, self.image_size, self.image_size], dtype=np.float32)
            sdf_gt = np.ones((self.image_size, self.image_size), dtype=np.float32)
        else:
            bbox = np.array([np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)], dtype=np.float32)
            dist_in = distance_transform_edt(mask_bin)
            dist_out = distance_transform_edt(1 - mask_bin)
            sdf_gt = (dist_out - dist_in).astype(np.float32)

        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        image_norm = (image_tensor - self.pixel_mean) / self.pixel_std
        return image_norm, torch.tensor(sdf_gt).unsqueeze(0), torch.tensor(bbox)

# ==========================================
# TRAINING LOOP
# ==========================================
def main():
    use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        print("🛠️ Single-GPU / Debug Mode Activated.")
        
    device = torch.device(f"cuda:{local_rank}")
    model = MedSAMPINN()
    
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        if local_rank == 0: print(f"✅ Phase 2 Weights Loaded from {CHECKPOINT_PATH}")
    else:
        if local_rank == 0: print(f"⚠️ Warning: Could not find {CHECKPOINT_PATH}. Training from scratch.")
    
    model.to(device)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    dataset = KaggleNerveTrainDataset(TRAIN_DIR)
    sampler = DistributedSampler(dataset) if use_ddp else None
    
    # drop_last=True PREVENTS DDP DESYNC CRASHES
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler('cuda')

    for epoch in range(EPOCHS):
        if use_ddp: sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(dataloader, disable=(local_rank != 0))
        
        for images, sdf_gts, bboxes in pbar:
            images, sdf_gts, bboxes = images.to(device), sdf_gts.to(device), bboxes.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                sdf_pred = model(images, bboxes)
                mse_loss = F.mse_loss(sdf_pred, sdf_gts)
                
                dy = sdf_pred[:, :, 1:, :] - sdf_pred[:, :, :-1, :]
                dx = sdf_pred[:, :, :, 1:] - sdf_pred[:, :, :, :-1]
                grad_norm = torch.sqrt(dx[:, :, :-1, :]**2 + dy[:, :, :, :-1]**2 + 1e-8)
                eikonal_loss = torch.mean((grad_norm - 1.0)**2)
                
                loss = mse_loss + 0.1 * eikonal_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if local_rank == 0: 
                pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

        if local_rank == 0: 
            torch.save(model.module.state_dict() if use_ddp else model.state_dict(), SAVE_PATH)

    if use_ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()