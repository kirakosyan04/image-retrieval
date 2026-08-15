# Image Retrieval with CLIP and FAISS

An image retrieval system that uses OpenAI's CLIP model to find images based on natural-language text queries.

## How it works

1. CIFAR-10 images are loaded.
2. CLIP converts images into 512-dimensional embeddings.
3. FAISS indexes the embeddings for fast similarity search.
4. A text query is converted into a CLIP text embedding.
5. FAISS retrieves the most similar images.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- CLIP
- FAISS
- CIFAR-10
- NumPy

## Project Structure

```text
image-retrieval/
├── src/
│   └── main.py
├── README.md
├── .gitignore
└── requirements.txt

## Goal

The goal of this project is to build a practical semantic image search system using multimodal embeddings.