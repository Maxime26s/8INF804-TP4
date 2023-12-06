from datetime import datetime
import json
import os
import torch
from torchvision.utils import save_image


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
    os.makedirs(os.path.join(args.model_path, "images"), exist_ok=True)
    for epoch in range(args.current_epoch + 1, args.n_epochs + 1):
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

        current_time = datetime.now()
        print(
            f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [Epoch {epoch}/{args.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
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
