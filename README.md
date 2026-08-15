# Multimodal Image Retrieval with CLIP

A multimodal image retrieval system that uses a pretrained CLIP model to retrieve visually and semantically relevant images from natural-language queries.

## Overview

The system converts images and text into a shared 512-dimensional embedding space using CLIP. It then uses cosine similarity to retrieve the most relevant images for a given text query.

The retrieval database contains 10,000 CIFAR-10 training images, while the separate 10,000-image test set is used to evaluate retrieval performance on previously unseen images.

## Pipeline

```text
Text Query
    ↓
CLIP Text Encoder
    ↓
512D Text Embedding
    ↓
Cosine Similarity
    ↓
Top-K Image Retrieval
```

```text
Training Images
    ↓
CLIP Vision Encoder
    ↓
512D Image Embeddings
    ↓
Retrieval Database
```

## Features

* Natural-language image search
* CLIP-based multimodal embeddings
* 512-dimensional image representations
* Cosine similarity retrieval
* Top-K image ranking
* Visual result grid
* Image-to-image retrieval evaluation
* Recall@K and Precision@K metrics

## Dataset

**CIFAR-10**

* 10,000 training images are used as the retrieval database.
* 10,000 test images are used as evaluation queries.
* The test images are not included in the retrieval database.

## Evaluation

The system evaluates image retrieval using all 10,000 CIFAR-10 test images as queries.

Each test image is converted into a CLIP embedding and used to retrieve the 10 most similar images from the 10,000-image training database.

### Results

| Metric       |  Score |
| ------------ | -----: |
| Recall@1     | 89.21% |
| Recall@5     | 96.84% |
| Recall@10    | 98.47% |
| Precision@10 | 86.85% |

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* CLIP
* NumPy
* Matplotlib
* CIFAR-10

## Example

Query:

```text
"a car"
```

The system converts the text query into a CLIP embedding and retrieves the top 10 most relevant images from the retrieval database, displaying their similarity scores and visual results.

## Project Structure

```text
image-retrieval/
├── src/
│   └── main.py
├── README.md
├── .gitignore
└── requirements.txt
```
