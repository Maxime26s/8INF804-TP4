import argparse
import json
import os
import torch

from src.GAN import Discriminator, Generator
from src.Training import train_and_evaluate_gan
from src.Utils import setup_data_loaders, display_generated_images, display_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
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
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument(
        "--img_size", type=int, default=64, help="Image size (width/height)"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of image channels"
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
        choices=["train", "eval"],
        help="Mode: train or eval",
    )
    parser.add_argument(
        "--current_epoch",
        type=int,
        default=0,
        help="Current epoch (for loading saved models)",
    )
    return parser.parse_args()


def initialize_gan_model(latent_dim, img_shape, normalization_choice, device):
    generator = Generator(img_shape, latent_dim, normalization_choice).to(device)
    discriminator = Discriminator(img_shape, normalization_choice).to(device)
    return generator, discriminator


if __name__ == "__main__":
    args = parse_args()
    print("Arguments parsed successfully.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_shape = (args.channels, args.img_size, args.img_size)
    print(f"Image shape set to: {img_shape}")

    generator, discriminator = initialize_gan_model(
        args.latent_dim, img_shape, args.normalization_choice, device
    )
    print("Initialized generator and discriminator models.")

    # Load model parameters if required
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
                args.latent_dim = saved_params.get("latent_dim", args.latent_dim)
                args.img_size = saved_params.get("img_size", args.img_size)
                args.normalization_choice = saved_params.get(
                    "normalization_choice", args.normalization_choice
                )
                args.channels = saved_params.get("channels", args.channels)
                args.current_epoch = saved_params.get(
                    "current_epoch", args.current_epoch
                )

            generator_path = os.path.join(
                args.model_path,
                f"generator_epoch_{args.current_epoch}.pt",
            )
            discriminator_path = os.path.join(
                args.model_path,
                f"discriminator_epoch_{args.current_epoch}.pt",
            )
            print(discriminator_path)

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
            print(
                "Saved model parameters not found. Starting training with default parameters."
            )

    generator.to(device)
    discriminator.to(device)
    print("Moved models to the designated device.")

    dataloader = setup_data_loaders(args.batch_size, args.img_size, "./images")
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
        train_and_evaluate_gan(
            generator,
            discriminator,
            dataloader,
            optimizer_G,
            optimizer_D,
            adversarial_loss,
            device,
            args,
            100,
        )
    elif args.mode == "eval":
        print("Entering evaluation mode...")
        generator.eval()

        # Fetch a batch of real images
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)

        # Generate noise vector and produce fake images
        z = torch.randn(real_images.size(0), args.latent_dim, device=device)
        gen_imgs = generator(z)

        # Display real images
        print("Displaying real images...")
        display_images(real_images, grayscale=(args.channels == 1))

        # Display generated images
        print("Displaying generated images...")
        display_generated_images(gen_imgs, grayscale=(args.channels == 1))
