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
                (imgs.shape[0], args.latent_size), device=device, dtype=torch.float
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
                "latent_size": args.latent_size,
                "img_size": args.img_size,
                "current_epoch": epoch,
                "normalization_choice": args.normalization_choice,
                "n_channels": args.n_channels,
                "n_epochs": args.n_epochs,
                "gan_type": args.gan_type,
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


def train_and_evaluate_dcgan(
    generator,
    discriminator,
    dataloader,
    optimizerG,
    optimizerD,
    criterion,
    device,
    args,
    sample_interval,
):
    g_losses = []
    d_losses = []

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(args.img_size, args.latent_size, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(args.current_epoch + 1, args.n_epochs + 1):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.latent_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Save Losses for plotting later
            g_losses.append(errG.item())
            d_losses.append(errD.item())

            if i == 0:
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"{time}\t[{epoch}/{args.n_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )

        if epoch % sample_interval == 0 or epoch == args.n_epochs:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                save_image(
                    fake.data[:25],
                    f"{args.model_path}/images/{epoch}.png",
                    nrow=5,
                    normalize=True,
                )

            current_params = {
                "batch_size": args.batch_size,
                "lr": args.lr,
                "latent_size": args.latent_size,
                "img_size": args.img_size,
                "current_epoch": epoch,
                "normalization_choice": args.normalization_choice,
                "n_channels": args.n_channels,
                "n_epochs": args.n_epochs,
                "gan_type": args.gan_type,
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
