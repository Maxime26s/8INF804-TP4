import argparse
from torchvision.utils import save_image
import torch
import os
import json

from src.GAN import Discriminator, Generator
from src.Utils import setup_data_loaders, display_generated_images


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


def train_and_evaluate_gan(
    generator,
    discriminator,
    dataloader,
    optimizer_G,
    optimizer_D,
    adversarial_loss,
    device,
    args,
    sample_interval,
):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            valid = torch.ones((imgs.size(0), 1), device=device, dtype=torch.float)
            fake = torch.zeros((imgs.size(0), 1), device=device, dtype=torch.float)

            # Configure input
            real_imgs = imgs.to(device).type(torch.float)

            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            z = torch.randn(
                (imgs.shape[0], args.latent_dim), device=device, dtype=torch.float
            )
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(
                    gen_imgs.data[:25],
                    f"{args.model_path}/images/{batches_done}.png",
                    nrow=5,
                    normalize=True,
                )

        print(
            f"[Epoch {epoch}/{args.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
        )

        if epoch % sample_interval == 0:
            current_params = {
                "batch_size": args.batch_size,
                "lr": args.lr,
                "latent_dim": args.latent_dim,
                "img_size": args.img_size,
                "current_epoch": epoch,
                "normalization_choice": args.normalization_choice,
                "channels": args.channels,
            }
            with open(f"{args.model_path}/parameters_dict.json", "w") as file:
                json.dump(current_params, file)

            torch.save(
                generator.state_dict(), f"{args.model_path}/generator_epoch_{epoch}.pt"
            )
            torch.save(
                discriminator.state_dict(),
                f"{args.model_path}/discriminator_epoch_{epoch}.pt",
            )


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_shape = (args.channels, args.img_size, args.img_size)

    generator, discriminator = initialize_gan_model(
        args.latent_dim, img_shape, args.normalization_choice, device
    )

    # Load model parameters if required
    if args.load_model:
        if os.path.exists(args.model_path):
            paremeters_path = os.path.join(args.model_path, "parameters_dict.json")
            if os.path.exists(paremeters_path):
                with open(paremeters_path, "r") as file:
                    saved_params = json.load(file)

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
                f"generateur_epoch_{args.current_epoch}.pt",
            )
            discriminator_path = os.path.join(
                args.model_path,
                f"discriminateur_epoch_{args.current_epoch}.pt",
            )

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

    dataloader = setup_data_loaders(args.batch_size, args.img_size, "./images")

    if args.mode == "train":
        # Training mode
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
        # Evaluation mode (image generation)
        generator.eval()
        z = torch.randn(args.batch_size, args.latent_dim, device=device)
        gen_imgs = generator(z).to(device)
        display_generated_images(gen_imgs, grayscale=(args.channels == 1))
