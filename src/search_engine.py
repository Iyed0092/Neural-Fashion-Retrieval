import torch
import faiss
import pickle
from transformers import CLIPProcessor, CLIPModel

class SearchEngine:
    def __init__(self, index_path, meta_path, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Search Engine on {self.device}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.index = faiss.read_index(index_path)
        
        with open(meta_path, 'rb') as f:
            self.filenames = pickle.load(f)

    def _get_embedding(self, inputs):
        """Helper to get normalized embeddings for text or image"""
        with torch.no_grad():
            if 'pixel_values' in inputs:
                outputs = self.model.get_image_features(**inputs)
            else:
                outputs = self.model.get_text_features(**inputs)
            
            # Normalize for cosine similarity
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
        return outputs.cpu().numpy().astype('float32')

    def search_text(self, query, k=10):
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        vector = self._get_embedding(inputs)
        return self._run_search(vector, k)

    def search_image(self, image, k=10):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        vector = self._get_embedding(inputs)
        return self._run_search(vector, k)

    def _run_search(self, vector, k):
        distances, indices = self.index.search(vector, k)
        
        results = []
        # FAISS returns batch results, we only need the first query
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.filenames):
                results.append((self.filenames[idx], float(dist)))
        
        return results