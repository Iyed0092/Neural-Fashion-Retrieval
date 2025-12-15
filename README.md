# 🛍️ Neural Fashion Retrieval
### A Multimodal Semantic Search Engine using CLIP & FAISS

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)
![OpenAI](https://img.shields.io/badge/OpenAI-CLIP-000000?style=for-the-badge&logo=openai)
![FAISS](https://img.shields.io/badge/Meta-FAISS-0078D4?style=for-the-badge&logo=meta)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)

</div>

---

## 🚀 Overview

**Neural Fashion Retrieval** is an AI-powered search engine capable of understanding the *semantic meaning* of apparel images. Unlike traditional keyword search, this system allows users to find products based on **visual similarity** (Texture, Cut, Style) or **natural language descriptions** (e.g., *"floral summer dress"*).

Built on the **DeepFashion** dataset, the system leverages **OpenAI's CLIP (ViT-B/32)** to project images and text into a shared high-dimensional vector space, indexed by **Meta's FAISS** for millisecond-latency retrieval.

(For a Faster execution please run it on gpu because the embeddings can take long time on cpu. Here I did use my RTX 4060 8GB Laptop

## ✨ Key Features

* **⚡ Zero-Shot Multimodal Search:** Query the database using Text-to-Image or Image-to-Image without training custom labels.
* **🧠 Visual Semantics:** Captures fine-grained details like "denim texture," "V-neck cut," or "vintage style" that keywords miss.
* **🏎️ High-Performance Indexing:** Uses **FAISS IVF-Flat** (Inverted File Index) to scale to thousands of items with <50ms query time.
* **📊 Proven Accuracy:** Validated against ground-truth metadata with a **Recall@10 of 96.8%**.
* **🎨 Interactive Dashboard:** A production-ready Streamlit interface for real-time visual exploration.

---

## 🏗️ Architecture

The system follows a standard **Dense Retrieval** pipeline:

1.  **Ingestion:** Images are processed via **CLIP (ViT-B/32)** to extract 512-dimensional feature vectors.
2.  **Indexing:** Vectors are normalized and stored in an **Inverted File Index (IVF)** using **FAISS** for efficient approximate nearest neighbor search.
3.  **Inference:**
    * *Text Query:* Converted to a vector via CLIP Text Encoder.
    * *Image Query:* Converted via CLIP Image Encoder.
4.  **Retrieval:** The query vector is compared against the index using **Cosine Similarity**.


## 📂 Project Structure

```bash
neural_fashion_search/
├── data/
│   ├── images/               # Raw fashion images (DeepFashion/Kaggle)
│   ├── embeddings/           # Generated .npy vectors & .pkl filenames
│   ├── faiss_index/          # Persisted FAISS index (.bin)
│   └── styles.csv            # Metadata for evaluation
│
├── src/
│   ├── feature_extractor.py  # ETL: Loads images -> CLIP -> Embeddings
│   ├── indexer.py            # Builds & saves the FAISS IVF index
│   ├── search_engine.py      # Core logic for query processing
│   └── evaluate.py           # Script to calculate Recall@K metrics
│
├── app.py                    # Streamlit Dashboard entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation

```
## 📊 Performance & Evaluation

The system was evaluated on a random subset of **1,000 queries** against the dataset metadata (`articleType`) to measure retrieval relevance.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Recall@1** | **84.30%** | The exact category match is the #1 result. |
| **Recall@5** | **94.90%** | The correct category appears in the top 5 results. |
| **Recall@10** | **96.80%** | The correct category appears in the top 10 results. |
| **Latency** | **< 50ms** | Average query time on a standard CPU. |

> *Evaluation run on `src/evaluate.py` using `styles.csv` as ground truth.*

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* GPU recommended (NVIDIA RTX) for fast extraction, but runs on CPU. (I did use for this project my RTX 4060 8GB Laptop)

### 1. Clone the Repository
```bash
git clone [https://github.com/Iyed0092/neural_fashion_search.git](https://github.com/Iyed0092/neural_fashion_search.git)
cd neural_fashion_search
```
## 💻 Usage Pipeline

### Step 1: Extract Features
Scans data/images, runs CLIP inference, and saves vectors.
``` bash
python src/feature_extractor.py
```

### Step 2: Build Index
Reads the vectors and compiles the optimized FAISS index.
``` bash
python src/indexer.py
```
### Step 3: Run the Dashboard
Reads the vectors and compiles the optimized FAISS index.
``` bash
streamlit run app.py
```
