import argparse
import json
import os
import random
import torch

import src.GAN as GAN
import src.DCGAN as DCGAN
from src.Training import (
    train_gan_bce,
    train_dcgan_bce,
    train_dcgan_wasserstein,
)
from src.Utils import (
    setup_data_loaders,
    display_generated_images,
    display_images,
    plot_acc,
    plot_loss,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning Rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="Beta 1 hyperparameter for Adam optimizers",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="Beta 2 hyperparameter for Adam optimizers",
    )
    parser.add_argument("--latent_size", type=int, default=64, help="Latent dimension")
    parser.add_argument(
        "--img_size", type=int, default=64, help="Image size (width/height)"
    )
    parser.add_argument(
        "--n_channels", type=int, default=1, help="Number of image channels"
    )
    parser.add_argument(
        "--normalization_choice", type=int, default=4, help="Normalization choice (0-5)"
    )
    parser.add_argument(
        "--load_model", type=int, default=0, help="Load saved model (0: No, 1: Yes)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models",
        help="Path to model folder",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "plot"],
        help="Mode: train, eval or plot",
    )
    parser.add_argument(
        "--current_epoch",
        type=int,
        default=0,
        help="Current epoch (for loading saved models)",
    )
    parser.add_argument(
        "--gan_type",
        type=str,
        default="gan",
        choices=["gan", "dcgan"],
        help="GAN type: gan or dcgan",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="bce",
        choices=["bce", "was"],
        help="Loss function: bce or was (Wasserstein Adversarial Loss)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def initialize_gan_model(img_shape, latent_size, normalization_choice, device):
    generator = GAN.Generator(img_shape, latent_size, normalization_choice).to(device)
    discriminator = GAN.Discriminator(img_shape, normalization_choice).to(device)
    return generator, discriminator


def initialize_dcgan_model(img_size, latent_size, n_channels, device):
    generator = DCGAN.Generator(img_size, latent_size, n_channels).to(device)
    generator.apply(DCGAN.weights_init)
    discriminator = DCGAN.Discriminator(img_size, n_channels).to(device)
    discriminator.apply(DCGAN.weights_init)
    return generator, discriminator


if __name__ == "__main__":
    args = parse_args()
    print("Arguments parsed successfully.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_shape = (args.n_channels, args.img_size, args.img_size)
    print(f"Image shape set to: {img_shape}")

    # Load model parameters if required
    history = {"g_losses": [], "d_losses": [], "d_accs": []}
    if args.load_model:
        print(f"Attempting to load model from {args.model_path}...")
        if os.path.exists(args.model_path):
            parameters_path = os.path.join(args.model_path, "parameters_dict.json")
            if os.path.exists(parameters_path):
                with open(parameters_path, "r") as file:
                    saved_params = json.load(file)

                print("Loaded saved model parameters.")

                args.batch_size = saved_params.get("batch_size", args.batch_size)
                args.lr = saved_params.get("lr", args.lr)
                args.latent_size = saved_params.get("latent_size", args.latent_size)
                args.img_size = saved_params.get("img_size", args.img_size)
                args.normalization_choice = saved_params.get(
                    "normalization_choice", args.normalization_choice
                )
                args.n_channels = saved_params.get("n_channels", args.n_channels)
                args.current_epoch = saved_params.get(
                    "current_epoch", args.current_epoch
                )
                args.n_epochs = saved_params.get("n_epochs", args.n_epochs)
                args.gan_type = saved_params.get("gan_type", args.gan_type)
                args.loss_function = saved_params.get(
                    "loss_function", args.loss_function
                )
                args.seed = saved_params.get("seed", args.seed)

                history = saved_params.get(
                    "history", {"g_losses": [], "d_losses": [], "d_accs": []}
                )
                history["g_losses"] = history.get("g_losses", [])
                history["d_losses"] = history.get("d_losses", [])
                history["d_accs"] = history.get("d_accs", [])

            set_seed(args.seed)
            fixed_noise = torch.randn(
                args.img_size, args.latent_size, 1, 1, device=device
            )

            generator_path = os.path.join(
                args.model_path,
                f"generator_epoch_{args.current_epoch}.pt",
            )
            discriminator_path = os.path.join(
                args.model_path,
                f"discriminator_epoch_{args.current_epoch}.pt",
            )

            if args.gan_type == "gan":
                generator, discriminator = initialize_gan_model(
                    img_shape, args.latent_size, args.normalization_choice, device
                )
            elif args.gan_type == "dcgan":
                generator, discriminator = initialize_dcgan_model(
                    args.img_size, args.latent_size, args.n_channels, device
                )
            print("Initialized generator and discriminator models.")

            if os.path.exists(generator_path) and os.path.exists(discriminator_path):
                generator.load_state_dict(
                    torch.load(generator_path, map_location=device)
                )
                discriminator.load_state_dict(
                    torch.load(discriminator_path, map_location=device)
                )
                print(f"Loaded model states from epoch {args.current_epoch}.")
            else:
                print(
                    "Saved model states not found for specified epoch. Starting training from scratch."
                )
        else:
            set_seed(args.seed)
            fixed_noise = torch.randn(
                args.img_size, args.latent_size, 1, 1, device=device
            )

            if args.gan_type == "gan":
                generator, discriminator = initialize_gan_model(
                    img_shape, args.latent_size, args.normalization_choice, device
                )
            elif args.gan_type == "dcgan":
                generator, discriminator = initialize_dcgan_model(
                    args.img_size, args.latent_size, args.n_channels, device
                )
            print("Initialized generator and discriminator models.")

            print(
                "Saved model parameters not found. Starting training with default parameters."
            )
    else:
        set_seed(args.seed)
        fixed_noise = torch.randn(args.img_size, args.latent_size, 1, 1, device=device)

        if args.gan_type == "gan":
            generator, discriminator = initialize_gan_model(
                img_shape, args.latent_size, args.normalization_choice, device
            )
        elif args.gan_type == "dcgan":
            generator, discriminator = initialize_dcgan_model(
                args.img_size, args.latent_size, args.n_channels, device
            )
        print("Initialized generator and discriminator models.")

    generator.to(device)
    discriminator.to(device)
    print("Moved models to the designated device.")

    dataloader = setup_data_loaders(
        args.batch_size, args.img_size, "./images", args.n_channels
    )
    print("Data loader setup complete.")

    if args.mode == "train":
        print("Entering training mode...")
        adversarial_loss = torch.nn.BCELoss()
        optimizer_G = torch.optim.Adam(
            generator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
        )
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
        )

        if args.gan_type == "gan":
            if args.loss_function == "bce":
                print("Training GAN with Binary Cross Entropy Loss...")
                train_gan_bce(
                    generator,
                    discriminator,
                    dataloader,
                    optimizer_G,
                    optimizer_D,
                    adversarial_loss,
                    device,
                    args,
                    history,
                    fixed_noise,
                    100,
                )
            elif args.loss_function == "was":
                print("Wasserstein Adversarial Loss not implemented for GANs.")
        elif args.gan_type == "dcgan":
            if args.loss_function == "bce":
                print("Training DCGAN with Binary Cross Entropy Loss...")
                train_dcgan_bce(
                    generator,
                    discriminator,
                    dataloader,
                    optimizer_G,
                    optimizer_D,
                    adversarial_loss,
                    device,
                    args,
                    history,
                    fixed_noise,
                    10,
                )
            elif args.loss_function == "was":
                print("Training DCGAN with Wasserstein Adversarial Loss...")
                train_dcgan_wasserstein(
                    generator,
                    discriminator,
                    dataloader,
                    optimizer_G,
                    optimizer_D,
                    device,
                    args,
                    history,
                    fixed_noise,
                    10,
                )
    elif args.mode == "eval":
        print("Entering evaluation mode...")
        generator.eval()

        # Fetch a batch of real images
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)

        # Generate noise vector and produce fake images
        z = torch.randn(real_images.size(0), args.latent_size, device=device)
        gen_imgs = generator(z)

        # Display real images
        print("Displaying real images...")
        display_images(real_images, grayscale=(args.n_channels == 1))

        # Display generated images
        print("Displaying generated images...")
        display_generated_images(gen_imgs, grayscale=(args.n_channels == 1))
    elif args.mode == "plot":
        print(f"Plotting results...")
        g_losses = history["g_losses"]
        d_losses = history["d_losses"]
        d_accs = history["d_accs"]
        epoch_count = args.current_epoch

        plot_loss(g_losses, d_losses, epoch_count)
        plot_acc(d_accs, epoch_count)
