import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import pickle
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "embeddings")

# config
BATCH_SIZE = 64
NUM_WORKERS = 4  
MODEL_NAME = "openai/clip-vit-base-patch32"

class FashionDataset(Dataset):
    def __init__(self, image_files, processor):
        self.image_files = image_files
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        path = os.path.join(IMAGE_DIR, file)
        
        try:
            # convert to RGB to ensure 3 channels
            image = Image.open(path).convert("RGB")
            processed = self.processor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0), file
        except Exception as e:
            # return None if image is corrupt
            return None, file

def collate_fn(batch):
    # filter out failed images (None)
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    
    pixel_values = torch.stack([item[0] for item in batch])
    filenames = [item[1] for item in batch]
    return pixel_values, filenames

def extract_features():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("No images found in data directory.")
        return

    print(f"Found {len(files)} images.")
    
    dataset = FashionDataset(files, processor)
    
    # dataloader handles parallel loading
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        pin_memory=True
    )

    embeddings = []
    filenames = []

    print("Starting extraction...")
    
    for pixels, batch_files in tqdm(loader):
        if pixels is None: continue

        pixels = pixels.to(device)

        with torch.no_grad():
            outputs = model.get_image_features(pixel_values=pixels)
            # normalize embeddings for cosine similarity
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
        embeddings.append(outputs.cpu().numpy().astype('float32'))
        filenames.extend(batch_files)

    if embeddings:
        full_embs = np.concatenate(embeddings)
        
        np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), full_embs)
        with open(os.path.join(OUTPUT_DIR, "filenames.pkl"), 'wb') as f:
            pickle.dump(filenames, f)
            
        print(f"Done. Saved {len(full_embs)} embeddings.")
    else:
        print("Failed to generate embeddings.")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    extract_features()