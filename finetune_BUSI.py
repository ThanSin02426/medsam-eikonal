import os
import sys
import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast, GradScaler

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    print("ERROR: scipy is missing. Run: pip install scipy")
    sys.exit(1)

from final_train import MedSAMPINN

# ==========================================
# CONFIGURATION - FIXED FOR MEDSAM COMPATIBILITY
# ==========================================
EPOCHS = 50
BATCH_SIZE = 1      # MUST BE 1 to avoid the MedSAM MaskDecoder batching bug
LEARNING_RATE = 5e-5 
LAMBDA_MSE = 1.0     
LAMBDA_EIKONAL = 0.1 
PHASE_1_WEIGHTS = "medsam_pinn_epoch_49.pth"

class ClinicalUltrasoundSDFDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=1024):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask > 127).astype(np.uint8)
        
        if mask_bin.max() == 0:
            bbox = np.array([0, 0, self.image_size, self.image_size], dtype=np.float32)
            sdf_gt = np.ones((self.image_size, self.image_size), dtype=np.float32) * 10.0
        else:
            y_indices, x_indices = np.where(mask_bin > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            noise = np.random.randint(-15, 15, size=4)
            bbox = np.array([
                max(0, x_min + noise[0]), max(0, y_min + noise[1]),
                min(self.image_size, x_max + noise[2]), min(self.image_size, y_max + noise[3])
            ], dtype=np.float32)

            dist_in = distance_transform_edt(mask_bin)
            dist_out = distance_transform_edt(1 - mask_bin)
            sdf_gt = (dist_out - dist_in).astype(np.float32)

        # Correct normalization for SAM image encoder
        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        image_normalized = (image_tensor - self.pixel_mean) / self.pixel_std

        return image_normalized, torch.tensor(sdf_gt).unsqueeze(0), torch.tensor(bbox)

def calc_eikonal_loss(pred_sdf):
    grad_x = (pred_sdf[:, :, 2:, 1:-1] - pred_sdf[:, :, :-2, 1:-1]) / 2.0
    grad_y = (pred_sdf[:, :, 1:-1, 2:] - pred_sdf[:, :, 1:-1, :-2]) / 2.0
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    return F.mse_loss(grad_norm, torch.ones_like(grad_norm))

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dataset = ClinicalUltrasoundSDFDataset("./Dataset_BUSI/images", "./Dataset_BUSI/masks")
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)

    model = MedSAMPINN()
    
    if os.path.exists(PHASE_1_WEIGHTS):
        state_dict = torch.load(PHASE_1_WEIGHTS, map_location=device, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            if not name.startswith('medsam.') and not name.startswith('sdf_proj'):
                name = 'medsam.' + name
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler('cuda')

    for epoch in range(EPOCHS):
        model.train()
        sampler.set_epoch(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(local_rank != 0))

        for images, sdf_gts, bboxes in pbar:
            images, sdf_gts, bboxes = images.to(device), sdf_gts.to(device), bboxes.to(device)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                sdf_preds = model(images, bboxes)
                loss_mse = F.mse_loss(sdf_preds, sdf_gts)
                loss_eikonal = calc_eikonal_loss(sdf_preds)
                total_loss = loss_mse + (LAMBDA_EIKONAL * loss_eikonal)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if local_rank == 0:
                pbar.set_postfix({"MSE": f"{loss_mse.item():.4f}", "Eik": f"{loss_eikonal.item():.4f}"})

        if local_rank == 0 and (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"medsam_pinn_busi_epoch_{epoch+1}.pth")

    dist.destroy_group()

if __name__ == "__main__":
    main()