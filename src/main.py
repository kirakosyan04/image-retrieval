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


print("\nEvaluating retrieval...")

test_queries = {
    "airplane": 0,
    "car": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}


correct = 0


for query, target_class in test_queries.items():

    top_indices, _ = search_images(
        query
    )

    top_classes = [
        dataset[idx][1]
        for idx in top_indices
    ]

    success = (
        target_class in top_classes
    )

    if success:
        correct += 1

    print(
        f"{query:10s} -> "
        f"{'PASS' if success else 'FAIL'}"
    )


recall_at_10 = (
    correct /
    len(test_queries)
)

print(
    f"\nRecall@10: "
    f"{recall_at_10:.2%}"
)
