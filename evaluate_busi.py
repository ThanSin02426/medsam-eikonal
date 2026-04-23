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
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from final_train import MedSAMPINN
from segment_anything import sam_model_registry

# ==========================================
# CHANGE THIS TO YOUR EXACT .PTH FILENAME IF DIFFERENT
# ==========================================
CUSTOM_PTH_FILE = "medsam_pinn_busi_epoch_50.pth" 

# ==========================================
# CLINICAL DATASET CLASS (EMBEDDED)
# ==========================================
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
            sdf_gt = np.ones((self.image_size, self.image_size), dtype=np.float32) * 1000.0 
            bbox = np.array([0, 0, self.image_size, self.image_size], dtype=np.float32)
        else:
            dist_in = distance_transform_edt(mask_bin)
            dist_out = distance_transform_edt(1 - mask_bin)
            sdf_gt = (dist_out - dist_in).astype(np.float32)
            y_indices, x_indices = np.where(mask_bin > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            noise = np.random.randint(-15, 15, size=4)
            bbox = np.array([
                max(0, x_min + noise[0]), max(0, y_min + noise[1]),
                min(self.image_size, x_max + noise[2]), min(self.image_size, y_max + noise[3])
            ], dtype=np.float32)

        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        image_normalized = (image_tensor - self.pixel_mean) / self.pixel_std

        return image_normalized, torch.tensor(sdf_gt).unsqueeze(0), torch.tensor(bbox)

# ==========================================
# EVALUATION LOGIC
# ==========================================
def compute_dice_torch(pred_mask, gt_mask, smooth=1e-5):
    pred = pred_mask.view(-1).float()
    gt = (gt_mask.view(-1) <= 0).float() 
    intersection = (pred * gt).sum()
    return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

def evaluate_busi():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    img_dir = "./Dataset_BUSI/images"
    mask_dir = "./Dataset_BUSI/masks"
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        if local_rank == 0:
            print(f"ERROR: Could not find {img_dir} or {mask_dir}.")
        dist.destroy_process_group()
        sys.exit(1)

    dataset = ClinicalUltrasoundSDFDataset(image_dir=img_dir, mask_dir=mask_dir)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2, pin_memory=True)

    if local_rank == 0:
        print(f"Total BUSI images to evaluate: {len(dataset)}")

    # ==========================================
    # PHASE A: EVALUATE BASELINE MEDSAM
    # ==========================================
    if local_rank == 0: print("\n--- Testing Baseline MedSAM on BUSI ---")
    base_model = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth").to(device)
    base_model.eval()

    base_dice_sum = torch.tensor(0.0, device=device)
    num_samples_local = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for images, sdf_gts, bboxes in tqdm(dataloader, desc="Baseline MedSAM", disable=(local_rank != 0)):
            images, sdf_gts, bboxes = images.to(device), sdf_gts.to(device), bboxes.to(device)
            
            with torch.amp.autocast('cuda'):
                img_emb = base_model.image_encoder(images)
                sparse_emb, dense_emb = base_model.prompt_encoder(
                    points=None, boxes=bboxes.unsqueeze(1), masks=None
                )
                low_res_masks, _ = base_model.mask_decoder(
                    image_embeddings=img_emb, image_pe=base_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                logits = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
                pred_binary = (logits > 0).float()
                
            base_dice_sum += compute_dice_torch(pred_binary, sdf_gts)
            num_samples_local += 1.0

    dist.all_reduce(base_dice_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_samples_local, op=dist.ReduceOp.SUM)
    final_base_dice = base_dice_sum / num_samples_local

    if local_rank == 0: print(f"✅ Baseline MedSAM BUSI Dice: {final_base_dice.item():.4f}\n")
    del base_model
    torch.cuda.empty_cache()

    # ==========================================
    # PHASE B: EVALUATE CUSTOM PHYSICS PINN
    # ==========================================
    if local_rank == 0: print(f"--- Testing Custom PINN ({CUSTOM_PTH_FILE}) on BUSI ---")
    pinn_model = MedSAMPINN()
    
    # Robust Dictionary Loading to fix Key Errors
    state_dict = torch.load(CUSTOM_PTH_FILE, map_location=device, weights_only=True)
    new_state_dict = {}
    
    for k, v in state_dict.items():
        # Strip DDP 'module.' prefix if it exists
        name = k[7:] if k.startswith('module.') else k
        
        # Add 'medsam.' prefix if it belongs to the wrapped model
        if not name.startswith('medsam.') and not name.startswith('sdf_proj'):
            name = 'medsam.' + name
            
        new_state_dict[name] = v

    pinn_model.load_state_dict(new_state_dict, strict=False)
    pinn_model.to(device)
    pinn_model.eval()

    pinn_dice_sum = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for images, sdf_gts, bboxes in tqdm(dataloader, desc="Custom Physics PINN", disable=(local_rank != 0)):
            images, sdf_gts, bboxes = images.to(device), sdf_gts.to(device), bboxes.to(device)
            
            with torch.amp.autocast('cuda'):
                sdf_pred = pinn_model(images, bboxes)
                pred_binary = (sdf_pred < 0).float()
            pinn_dice_sum += compute_dice_torch(pred_binary, sdf_gts)

    dist.all_reduce(pinn_dice_sum, op=dist.ReduceOp.SUM)
    final_pinn_dice = pinn_dice_sum / num_samples_local

    if local_rank == 0:
        print(f"✅ Custom PINN BUSI Dice: {final_pinn_dice.item():.4f}\n")
        print("==========================================")
        print("🏆 FINAL BUSI DATASET COMPARISON")
        print("==========================================")
        print(f"Baseline MedSAM:     {final_base_dice.item():.4f}")
        print(f"Custom Physics PINN: {final_pinn_dice.item():.4f}")
        print("==========================================")

    dist.destroy_process_group()

if __name__ == "__main__":
    evaluate_busi()