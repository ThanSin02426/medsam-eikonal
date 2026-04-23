import os
import sys
import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from tqdm import tqdm
from torch.amp import autocast

from final_train import MedSAMPINN
from segment_anything import sam_model_registry

# ==========================================
# CONFIGURATION
# ==========================================
CUSTOM_PTH_FILE = "medsam_pinn_busi_epoch_50.pth" 
TRAIN_DATA_DIR = "./train" 

# ==========================================
# KAGGLE NERVE DATASET CLASS (POSITIVE CASES ONLY)
# ==========================================
class NerveUltrasoundDataset(Dataset):
    def __init__(self, data_dir, image_size=1024):
        self.data_dir = data_dir
        self.image_size = image_size
        
        all_files = sorted(os.listdir(data_dir))
        raw_image_files = [f for f in all_files if f.endswith('.tif') and '_mask' not in f]
        
        self.image_files = []
        
        # PRE-FILTER: Keep only images where the nerve actually exists
        print(f"Scanning {len(raw_image_files)} images for positive nerve cases...")
        for img_filename in raw_image_files:
            mask_filename = img_filename.replace('.tif', '_mask.tif')
            mask_path = os.path.join(self.data_dir, mask_filename)
            
            # Fast check if mask has any white pixels
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.max() > 0:
                self.image_files.append(img_filename)
                
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        mask_filename = img_filename.replace('.tif', '_mask.tif')
        
        img_path = os.path.join(self.data_dir, img_filename)
        mask_path = os.path.join(self.data_dir, mask_filename)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        mask_bin = (mask > 127)
        mask_bin = binary_fill_holes(mask_bin).astype(np.uint8)
        
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

        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        image_normalized = (image_tensor - self.pixel_mean) / self.pixel_std

        return image_normalized, torch.tensor(sdf_gt).unsqueeze(0), torch.tensor(bbox)

# ==========================================
# EVALUATION LOGIC
# ==========================================
def compute_dice_torch(pred_mask, gt_mask, smooth=1e-5):
    pred = (pred_mask > 0).float()
    gt = (gt_mask.view(-1) <= 0).float() 
    
    # Special case: Both are completely empty (Correct prediction of no nerve)
    if pred.sum() == 0 and gt.sum() == 0:
        return torch.tensor(1.0, device=pred.device)
        
    intersection = (pred.view(-1) * gt).sum()
    return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

def evaluate_nerve():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dataset = NerveUltrasoundDataset(data_dir=TRAIN_DATA_DIR)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2, pin_memory=True)

    if local_rank == 0:
        print(f"Total Kaggle Nerve images: {len(dataset)}")

    if local_rank == 0: print("\n--- Testing Baseline MedSAM on Nerves ---")
    base_model = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth").to(device)
    base_model.eval()
    base_dice_sum = torch.tensor(0.0, device=device)
    num_samples = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for images, sdf_gts, bboxes in tqdm(dataloader, disable=(local_rank != 0)):
            images, sdf_gts, bboxes = images.to(device), sdf_gts.to(device), bboxes.to(device)
            with autocast('cuda'):
                img_emb = base_model.image_encoder(images)
                sparse_emb, dense_emb = base_model.prompt_encoder(None, bboxes.unsqueeze(1), None)
                low_res_masks, _ = base_model.mask_decoder(img_emb, base_model.prompt_encoder.get_dense_pe(), sparse_emb, dense_emb, False)
                logits = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear")
                pred_binary = (logits > 0).float()
            base_dice_sum += compute_dice_torch(pred_binary, sdf_gts)
            num_samples += 1.0

    if local_rank == 0: print(f"\n--- Testing PINN (Trained on Breast) on Nerves ---")
    pinn_model = MedSAMPINN()
    state_dict = torch.load(CUSTOM_PTH_FILE, map_location=device, weights_only=True)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    pinn_model.load_state_dict(new_state_dict, strict=False)
    pinn_model.to(device).eval()
    pinn_dice_sum = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for images, sdf_gts, bboxes in tqdm(dataloader, disable=(local_rank != 0)):
            images, sdf_gts, bboxes = images.to(device), sdf_gts.to(device), bboxes.to(device)
            with autocast('cuda'):
                sdf_pred = pinn_model(images, bboxes)
                pred_binary = (sdf_pred < 0).float()
            pinn_dice_sum += compute_dice_torch(pred_binary, sdf_gts)

    dist.all_reduce(base_dice_sum); dist.all_reduce(pinn_dice_sum); dist.all_reduce(num_samples)
    
    if local_rank == 0:
        print("\n==========================================")
        print(" KAGGLE NERVE DATASET SCORES")
        print("==========================================")
        print(f"Baseline MedSAM: { (base_dice_sum/num_samples).item():.4f}")
        print(f"PINN (Zero-Shot): { (pinn_dice_sum/num_samples).item():.4f}")
        print("==========================================")

    dist.destroy_process_group()

if __name__ == "__main__":
    evaluate_nerve()