import torch
import matplotlib.pyplot as plt

#def create_random_labels_map(classes: int) -> dict[int, tuple[int, int, int]]:
def create_random_labels_map(classes):

    labels_map: Dict[int, Tuple[int, int, int]] = {}
    for i in classes:
        labels_map[i] = torch.randint(0, 255, (3,))
    labels_map[0] = torch.zeros(3)
    return labels_map

#def labels_to_image(img_labels: torch.Tensor, labels_map: dict[int, tuple[int, int, int]]) -> torch.Tensor:
def labels_to_image(img_labels, labels_map):

    """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
    representation for visualisation purposes."""
    assert len(img_labels.shape) == 2, img_labels.shape
    H, W = img_labels.shape
    out = torch.empty(3, H, W, dtype=torch.uint8)
    for label_id, label_val in labels_map.items():
        mask = img_labels == label_id
        for i in range(3):
            out[i].masked_fill_(mask, label_val[i])
    return out

def show_components(img, labels):
    color_ids = torch.unique(labels)
    labels_map = create_random_labels_map(color_ids)
    labels_img = labels_to_image(labels, labels_map)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    # Showing Original Image
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Orginal Image")

    # Showing Image after Component Labeling
    ax2.imshow(labels_img.permute(1, 2, 0).squeeze().numpy())
    ax2.axis("off")
    ax2.set_title("Component Labeling")

    #plt.show()
    return plt
