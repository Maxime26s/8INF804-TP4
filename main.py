import argparse
import os
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch

from GAN import Discriminator, Generator

if __name__ == "__main__":
    # %% HYPERPARAMETRES

    PATH = "./images"  # chemin vers les images

    path_to_model = "./models"
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)  # Créer le répertoire

    n_epochs = 10000
    batch_size = 2048
    lr = 0.0002
    latent_dim = 64
    img_size = 64
    current_epoch = 0
    channels = 1

    normalization_choice = 4
    # 0 -> Nothing
    # 1 -> only spectral normalization on discriminator
    # 2 -> only batch_norm on generator
    # 3 -> only instance normalization on generator
    # 4 -> both spectral and batch
    # 5 -> both spectral and instance

    parameters_dict = {
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "latent_dim": latent_dim,
        "img_dim": img_size,
        "current_epoch": current_epoch,
        "normalization_choice": normalization_choice,
        "channels": channels,
    }

    # Si vous voulez charger un modèle, loaded_model doit valeur 1 et path_loaded_model doit être le chemin vers la sauvegarde du modèle
    loaded_model = 0
    path_parameters_dict = path_to_model + "/" + "parameters_dict.json"
    if loaded_model == 1:
        with open(path_parameters_dict, "r") as file:
            data = json.load(file)
            parameters_dict = dict(data)
            n_epochs = parameters_dict["n_epochs"]
            batch_size = parameters_dict["batch_size"]
            lr = parameters_dict["lr"]
            latent_dim = parameters_dict["latent_dim"]
            img_size = parameters_dict["img_dim"]
            current_epoch = parameters_dict["current_epoch"]
            normalization_choice = parameters_dict["normalization_choice"]
            channels = parameters_dict["channels"]

    # A priori, pas besoin de modifier ces paramètres
    os.makedirs("images", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=n_epochs, help="number of epochs of training"
    )  # 200
    parser.add_argument(
        "--batch_size", type=int, default=batch_size, help="size of the batches"
    )  # 64
    parser.add_argument("--lr", type=float, default=lr, help="adam: learning rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=latent_dim,
        help="dimensionality of the latent space",
    )
    parser.add_argument(
        "--img_size", type=int, default=img_size, help="size of each image dimension"
    )  # 28
    parser.add_argument(
        "--channels", type=int, default=1, help="number of image channels"
    )  # 1
    parser.add_argument(
        "--sample_interval", type=int, default=400, help="interval betwen image samples"
    )
    opt = parser.parse_args("")
    print(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # %% PRE-ENTRAINEMENT

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(img_shape, latent_dim, normalization_choice)
    discriminator = Discriminator(img_shape, normalization_choice)

    if loaded_model == 1:
        pathd = path_to_model + "/" + "discriminateur.pt"
        pathg = path_to_model + "/" + "generator.pt"
        # Discriminator
        discriminator.load_state_dict(torch.load(pathd))
        discriminator.train()
        # Generator
        generator.load_state_dict(torch.load(pathg))
        generator.train()

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    adversarial_loss = adversarial_loss.to(device)

    # Transformation applied
    transform = transforms.Compose(
        [
            transforms.Resize(
                (opt.img_size, opt.img_size)
            ),  # resize images at opt.img_size
            transforms.Grayscale(),  # Grayscale
            transforms.ToTensor(),  # images -> tensor
            transforms.Normalize([0.5], [0.5]),  # normalise tensors
        ]
    )

    # Download of images
    dataset = datasets.ImageFolder(root=PATH, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    # %% ENTRAINEMENT

    D_losses = []
    G_losses = []
    iterations = []

    # Chargement des courbes si loaded_model == 1
    if loaded_model == 1:
        # modifier en dur ces chemin pour charger un modèle précis
        pathg_cost = path_to_model + "/" + "generator_cost.json"
        pathd_cost = path_to_model + "/" + "discriminateur_cost.json"
        with open(pathg_cost, "r") as file:
            datag = json.load(file)
        with open(pathd_cost, "r") as file:
            datad = json.load(file)
        G_losses = list(datag)
        D_losses = list(datad)

    time_start = time.time()

    for epoch in range(1, opt.n_epochs + 1):
        for i, (imgs, _) in enumerate(dataloader):
            if (epoch % 50 == 0) and (i == 0) and (epoch > 0):  # speed evaluation
                time_spent = round(time.time() - time_start)
                average_speed = round(epoch / time_spent, 2)
                estimated_time = round((opt.n_epochs - epoch) / average_speed)
                time_formate = time.strftime("%Hh %Mmin %Ss", time.gmtime(time_spent))
                estimated_formate = time.strftime(
                    "%Hh %Mmin %Ss", time.gmtime(estimated_time)
                )
                print(
                    f"\033[91m Epoch: {epoch}/{opt.n_epochs}, time spent: {time_formate}, average speed until now: {average_speed} epoch/s and estimated time remaining : {estimated_formate}.\033[0m"
                )

            # Adversarial ground truths
            valid = torch.ones((imgs.size(0), 1), dtype=torch.float, device=device)
            fake = torch.zeros((imgs.size(0), 1), dtype=torch.float, device=device)

            # Configure input
            real_imgs = imgs.to(device).type(torch.float)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(
                (imgs.shape[0], opt.latent_dim), dtype=torch.float, device=device
            )

            # Generate a batch of images
            gen_imgs = generator(z, img_shape)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 200 == 0:
                current_time = datetime.now()
                log_entry = f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}"
                log_entry += f" [Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                print(log_entry)

            iterations.append(i)
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(
                    gen_imgs.data[:25],
                    "./images/%d.png" % batches_done,
                    nrow=5,
                    normalize=True,
                )

        D_losses.append(d_loss.item())
        G_losses.append(g_loss.item())
        if epoch % 1000 == 0:
            pathd = path_to_model + "/" + f"discriminateur_epoch_{epoch}.pt"
            pathg = path_to_model + "/" + f"generator_epoch_{epoch}.pt"
            torch.save(generator.state_dict(), pathg)
            torch.save(discriminator.state_dict(), pathd)
            print("D and G saved")
            # Sauvegarde du modèle

    # %% GENERATION

    # Enregistrement du Gen et du Disc
    pathd = path_to_model + "/" + "discriminateur.pt"
    pathg = path_to_model + "/" + "generator.pt"
    torch.save(generator.state_dict(), pathg)
    torch.save(discriminator.state_dict(), pathd)

    # Enregistrement des courbes
    pathd_cost = path_to_model + "/" + "discriminateur_cost.json"
    pathg_cost = path_to_model + "/" + "generator_cost.json"
    with open(pathg_cost, "w") as file:
        json.dump(G_losses, file)
    with open(pathd_cost, "w") as file:
        json.dump(D_losses, file)

    # Enregistrement des paramètres
    path_parameters_dict = path_to_model + "/" + "parameters_dict.json"
    with open(path_parameters_dict, "w") as file:
        json.dump(parameters_dict, file)

    # Appel du générateur
    generator_version = "/generator_epoch_4000.pt"
    generator = Generator(img_shape, latent_dim, normalization_choice)
    generator.load_state_dict(
        torch.load(path_to_model + generator_version, map_location=device)
    )
    generator.eval()

    (imgs, _) = next(iter(dataloader))
    generator.train()  # netD.eval() pour le après l'entraînement
    Gray = 1  # 0 for gray, 1 for RGB

    # Génération des images à parit d'un bruit gaussien
    z = torch.randn(imgs.shape[0], opt.latent_dim)
    gen_imgs = generator(z, img_shape)

    # %% AFFICHAGE DES FIGURES

    # Affichage des fonctions coûts
    plt.plot(D_losses, "g")
    plt.plot(G_losses, "b")
    plt.legend(["Discriminateur", "Générateur"])
    plt.show()

    # Real images
    batch = next(iter(dataloader))
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = batch[0][i]
        if Gray:
            img_gray = torch.mean(img, dim=0, keepdim=True)
            ax.imshow(img_gray.squeeze(), cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")
    plt.suptitle("Real Images")
    plt.show()

    # Generated images
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if Gray:
            img = gen_imgs[i].permute(1, 2, 0).detach()
            img_gray = torch.mean(img, dim=2, keepdim=True)
            ax.imshow(img_gray.squeeze(), cmap="gray")
        else:
            img = (
                gen_imgs[i].permute(1, 2, 0).detach().numpy()
            )  # Permuter les dimensions de l'image et détacher le tenseur
            ax.imshow(img)
        ax.axis("off")
    plt.suptitle("Generated images")
    plt.show()
