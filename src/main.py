import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
TOP_K = 10


print("Loading CLIP...")

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

model.eval()

print("CLIP loaded!")


print("Loading CIFAR-10...")

full_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True
)

dataset = torch.utils.data.Subset(
    full_dataset,
    range(10000)
)

print("Images:", len(dataset))


print("Generating image embeddings...")

image_embeddings = []

for start in tqdm(
    range(
        0,
        len(dataset),
        BATCH_SIZE
    )
):

    images = [
        dataset[i][0]
        for i in range(
            start,
            min(
                start + BATCH_SIZE,
                len(dataset)
            )
        )
    ]

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        image_features = model.visual_projection(
            vision_outputs.pooler_output
        )


    image_features = image_features / (
        image_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    image_embeddings.append(
        image_features.cpu().numpy()
    )


image_embeddings = np.vstack(
    image_embeddings
).astype("float32")

print(
    "Embedding matrix:",
    image_embeddings.shape
)

print("\nLoading CIFAR-10 test set...")

test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True
)

test_embeddings = []

print("Generating test image embeddings...")

for start in tqdm(
    range(
        0,
        len(test_dataset),
        BATCH_SIZE
    )
):

    images = [
        test_dataset[i][0]
        for i in range(
            start,
            min(
                start + BATCH_SIZE,
                len(test_dataset)
            )
        )
    ]

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        features = model.visual_projection(
            vision_outputs.pooler_output
        )

    features = features / (
        features.norm(
            dim=-1,
            keepdim=True
        )
    )

    test_embeddings.append(
        features.cpu().numpy()
    )


test_embeddings = np.vstack(
    test_embeddings
).astype("float32")

print(
    "Test embedding matrix:",
    test_embeddings.shape
)

def get_text_embedding(query):

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        text_features = model.text_projection(
            text_outputs.pooler_output
        )

    text_features = text_features / (
        text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    return (
        text_features
        .cpu()
        .numpy()
        .astype("float32")
    )[0]


def search_images(query):

    text_embedding = get_text_embedding(
        query
    )

    similarities = (
        image_embeddings @ text_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:TOP_K]

    return top_indices, similarities


def show_results(
    query,
    top_indices,
    similarities
):

    print("\nTop results:")

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(12, 5)
    )

    for rank, (ax, idx) in enumerate(
        zip(
            axes.flat,
            top_indices
        ),
        start=1
    ):

        image, label = dataset[idx]

        score = similarities[idx]

        ax.imshow(image)

        ax.set_title(
            f"{rank}. "
            f"{full_dataset.classes[label]}\n"
            f"Similarity: {score:.3f}"
        )

        ax.axis("off")

        print(
            f"{rank}. "
            f"Image #{idx} | "
            f"class={full_dataset.classes[label]} | "
            f"similarity={score:.4f}"
        )

    plt.suptitle(
        f'Query: "{query}"',
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


print("\nImage retrieval system ready!")

while True:

    query = input(
        "\nDescribe the image you want: "
    )

    if query.lower() in [
        "exit",
        "quit"
    ]:

        print("Goodbye!")

        break

    top_indices, similarities = search_images(
        query
    )

    show_results(
        query,
        top_indices,
        similarities
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print("\nEvaluating image retrieval...")

recall_at_1 = []
recall_at_5 = []
recall_at_10 = []
precision_at_10 = []

for query_idx in tqdm(range(len(test_dataset))):

    query_embedding = test_embeddings[query_idx]

    similarities = image_embeddings @ query_embedding

    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    query_class = test_dataset[query_idx][1]

    retrieved_classes = [
        dataset[idx][1]
        for idx in top_indices
    ]

    recall_at_1.append(
        int(query_class in retrieved_classes[:1])
    )

    recall_at_5.append(
        int(query_class in retrieved_classes[:5])
    )

    recall_at_10.append(
        int(query_class in retrieved_classes[:10])
    )

    correct = sum(
        cls == query_class
        for cls in retrieved_classes
    )

    precision_at_10.append(correct / 10)


print(f"Recall@1: {np.mean(recall_at_1):.2%}")
print(f"Recall@5: {np.mean(recall_at_5):.2%}")
print(f"Recall@10: {np.mean(recall_at_10):.2%}")
print(f"Precision@10: {np.mean(precision_at_10):.2%}")
