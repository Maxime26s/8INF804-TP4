import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def setup_data_loaders(batch_size, img_size, path_to_images, n_channels):
    if n_channels == 1:  # Grayscale images
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    elif n_channels == 3:  # Color images
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    dataset = datasets.ImageFolder(root=path_to_images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader


def display_images(images, grayscale=False):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).detach().cpu()
        if grayscale:
            img = img.mean(dim=2, keepdim=True)
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img.numpy())
        ax.axis("off")
    plt.suptitle("Real Images")
    plt.show()


def display_generated_images(gen_imgs, grayscale=False):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = gen_imgs[i].permute(1, 2, 0).detach().cpu()
        if grayscale:
            img_gray = torch.mean(img, dim=2, keepdim=True)
            ax.imshow(img_gray.squeeze(), cmap="gray")
        else:
            ax.imshow(img.numpy())
        ax.axis("off")
    plt.suptitle("Generated Images")
    plt.show()


def plot_loss(g_losses, d_losses, epoch_count):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss - EPOCH {}".format(epoch_count))
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_acc(d_accs, epoch_count):
    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Accuracy - EPOCH {}".format(epoch_count))
    plt.plot(d_accs, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim(0, 1)
    plt.show()
