import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.cuda.amp import autocast, GradScaler

from segment_anything import sam_model_registry

# ==========================================
# 1. SYNTHETIC DATASET & VIT NORMALIZATION
# ==========================================
class SyntheticUltrasoundSDFDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=1024):
        self.num_samples = num_samples
        self.image_size = image_size
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Random basic shape to simulate ROI
        center_x = np.random.randint(256, 768)
        center_y = np.random.randint(256, 768)
        radius = np.random.randint(50, 200)
        y, x = np.ogrid[:self.image_size, :self.image_size]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask[dist_from_center <= radius] = 1

        # Continuous Euclidean Distance
        dist_in = distance_transform_edt(mask)
        dist_out = distance_transform_edt(1 - mask)
        sdf_gt = dist_out - dist_in
        
        # Simulated BBox prompt with noise
        noise = np.random.randint(-10, 10, size=4)
        bbox = np.array([
            max(0, center_x - radius + noise[0]),
            max(0, center_y - radius + noise[1]),
            min(self.image_size, center_x + radius + noise[2]),
            min(self.image_size, center_y + radius + noise[3])
        ], dtype=np.float32)

        # Create dummy image and apply strict ViT normalization
        image = (mask * 200 + np.random.randint(0, 55, size=mask.shape)).astype(np.float32)
        image_tensor = torch.tensor(np.stack([image]*3, axis=0), dtype=torch.float32)
        image_normalized = (image_tensor - self.pixel_mean) / self.pixel_std

        return (
            image_normalized, 
            torch.tensor(sdf_gt, dtype=torch.float32).unsqueeze(0), 
            torch.tensor(bbox, dtype=torch.float32)
        )

# ==========================================
# 2. UPWIND EIKONAL LOSS (STABILIZED PHYSICS)
# ==========================================
class UpwindEikonalLoss(nn.Module):
    """
    Stabilized Eikonal Loss using an upwind-style finite difference scheme.
    Prevents artificial loss spikes at the zero-level set (object boundaries).
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, sdf_pred):
        # Replicate padding to compute accurate differences at image borders
        padded = F.pad(sdf_pred, (1, 1, 1, 1), mode='replicate')
        
        # Forward and backward differences for X axis
        dx_fwd = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, 1:-1]
        dx_bwd = padded[:, :, 1:-1, 1:-1] - padded[:, :, 1:-1, :-2]
        
        # Forward and backward differences for Y axis
        dy_fwd = padded[:, :, 2:, 1:-1] - padded[:, :, 1:-1, 1:-1]
        dy_bwd = padded[:, :, 1:-1, 1:-1] - padded[:, :, :-2, 1:-1]
        
        # Upwind logic: take the maximum absolute difference
        dx = torch.maximum(torch.abs(dx_fwd), torch.abs(dx_bwd))
        dy = torch.maximum(torch.abs(dy_fwd), torch.abs(dy_bwd))
        
        # Physics constraint: Magnitude of spatial gradient should equal 1
        grad_magnitude = torch.sqrt(dx**2 + dy**2 + self.epsilon)
        loss = torch.mean((grad_magnitude - 1.0)**2)
        return loss

# ==========================================
# 3. ARCHITECTURE: MEDSAM + PROJECTION HEAD
# ==========================================
class MedSAMPINN(nn.Module):
    """
    Wraps MedSAM, freezes encoders, and appends a projection head 
    to map binary logits to continuous Euclidean distances.
    """
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.medsam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        else:
            self.medsam = sam_model_registry["vit_b"]() # Fallback

        # Strictly freeze Encoders to save VRAM
        for param in self.medsam.image_encoder.parameters(): param.requires_grad = False
        for param in self.medsam.prompt_encoder.parameters(): param.requires_grad = False
        
        # Unfreeze Mask Decoder
        for param in self.medsam.mask_decoder.parameters(): param.requires_grad = True

        # Projection Head: Scales logits to SDF physical space
        self.sdf_proj = nn.Conv2d(1, 1, kernel_size=1)
        nn.init.constant_(self.sdf_proj.weight, 10.0)
        nn.init.constant_(self.sdf_proj.bias, 0.0)

    def forward(self, images, bboxes):
        with torch.no_grad():
            image_embeddings = self.medsam.image_encoder(images)
            sparse_emb, dense_emb = self.medsam.prompt_encoder(
                points=None,
                boxes=bboxes.unsqueeze(1),
                masks=None,
            )

        low_res_logits, _ = self.medsam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.medsam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False, 
        )

        low_res_sdf = self.sdf_proj(low_res_logits)
        sdf_pred = F.interpolate(low_res_sdf, size=(1024, 1024), mode="bilinear", align_corners=False)
        return sdf_pred

# ==========================================
# 4. DISTRIBUTED TRAINING LOOP WITH AMP
# ==========================================
def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # HYPERPARAMETERS - Optimized for < 24GB VRAM
    BATCH_SIZE = 1 # Keeping this safely at 1 to prevent OOM
    EPOCHS = 50
    LR = 1e-4
    LAMBDA_MSE = 1.0
    LAMBDA_EIKONAL = 0.5 

    dataset = SyntheticUltrasoundSDFDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)

    model = MedSAMPINN(checkpoint_path="medsam_vit_b.pth").to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    mse_criterion = nn.MSELoss()
    eikonal_criterion = UpwindEikonalLoss()
    
    # Initialize Mixed Precision Scaler
    scaler = GradScaler()

    model.train()
    
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        
        for step, (images, sdf_gts, bboxes) in enumerate(dataloader):
            images, sdf_gts, bboxes = images.to(device), sdf_gts.to(device), bboxes.to(device)
            
            optimizer.zero_grad()

            # Automatic Mixed Precision Forward Pass
            with autocast():
                sdf_pred = model(images, bboxes)
                loss_mse = mse_criterion(sdf_pred, sdf_gts)
                loss_eikonal = eikonal_criterion(sdf_pred)
                loss = (LAMBDA_MSE * loss_mse) + (LAMBDA_EIKONAL * loss_eikonal)
            
            # Scaled Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if local_rank == 0 and step % 10 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{step}/{len(dataloader)}] "
                      f"Total Loss: {loss.item():.4f} | MSE: {loss_mse.item():.4f} | Eikonal: {loss_eikonal.item():.4f}")

        if local_rank == 0:
            print(f"--- Epoch {epoch} Complete | Avg Loss: {epoch_loss/len(dataloader):.4f} ---")
            torch.save(model.module.state_dict(), f"medsam_pinn_epoch_{epoch}.pth")

    if local_rank == 0:
        print("Phase 1 Eikonal Training Complete.")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    train()