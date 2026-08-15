# Multimodal Image Retrieval with CLIP

A multimodal image retrieval system that uses OpenAI's CLIP model to retrieve relevant images from natural-language queries.

## Overview

The system converts both images and text into a shared 512-dimensional embedding space using CLIP. It then uses cosine similarity to retrieve the most relevant images for a given text query.

## Pipeline

Text Query
↓
CLIP Text Encoder
↓
512D Text Embedding
↓
Cosine Similarity
↓
Top-K Image Retrieval

Images
↓
CLIP Vision Encoder
↓
512D Image Embeddings

## Features

- Natural-language image search
- CLIP-based multimodal embeddings
- 512-dimensional image representations
- Cosine similarity retrieval
- Top-K image ranking
- Visual result grid
- Retrieval evaluation

## Dataset

CIFAR-10

10,000 images are used for the retrieval index.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- OpenAI CLIP
- NumPy
- Matplotlib
- CIFAR-10

## Example

Query:

"a car"

The system retrieves the most visually and semantically relevant images and displays the top 10 results with similarity scores.

## Project Structure

```text
image-retrieval/
├── src/
│   └── main.py
├── README.md
├── .gitignore
└── requirements.txt
 ``` 
