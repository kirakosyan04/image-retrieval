import torch
import numpy as np
import faiss

from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel


print("Loading CLIP...")

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()

print("CLIP loaded!")


print("Loading CIFAR-10...")

dataset = CIFAR10(
    root="data",
    train=True,
    download=True
)

images = dataset.data[:10000]
labels = np.array(dataset.targets[:10000])

print("Images:", len(images))


embeddings = []

print("Generating image embeddings...")

batch_size = 64

for start in tqdm(
    range(0, len(images), batch_size)
):

    batch_images = images[start:start + batch_size]

    pil_images = [
        transforms.ToPILImage()(img)
        for img in batch_images
    ]

    inputs = processor(
        images=pil_images,
        return_tensors="pt"
    )

    with torch.no_grad():
        image_features = model.get_image_features(
            **inputs
        )

    image_features = image_features / (
        image_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    embeddings.append(
        image_features.cpu().numpy()
    )


embeddings = np.vstack(embeddings).astype("float32")

print("Embedding matrix:", embeddings.shape)


dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index created!")
print("Indexed images:", index.ntotal)


query = input("\nDescribe the image you want: ")

text_inputs = processor(
    text=[query],
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    text_features = model.get_text_features(
        **text_inputs
    )

text_features = text_features / (
    text_features.norm(
        dim=-1,
        keepdim=True
    )
)

text_embedding = (
    text_features
    .cpu()
    .numpy()
    .astype("float32")
)


scores, indices = index.search(
    text_embedding,
    10
)


class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("\nTop results:\n")

for rank, (idx, score) in enumerate(
    zip(indices[0], scores[0]),
    start=1
):

    label = labels[idx]

    print(
        f"{rank}. "
        f"Image #{idx} | "
        f"class={class_names[label]} | "
        f"similarity={score:.4f}"
    )