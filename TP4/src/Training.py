from datetime import datetime
import json
import os
import torch
from torchvision.utils import save_image


def save_state(generator, discriminator, epoch, fixed_noise, args, history):
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
        "loss_function": args.loss_function,
        "history": history,
    }
    with open(f"{args.model_path}/parameters_dict.json", "w") as file:
        json.dump(current_params, file)

    torch.save(generator.state_dict(), f"{args.model_path}/generator_epoch_{epoch}.pt")
    torch.save(
        discriminator.state_dict(),
        f"{args.model_path}/discriminator_epoch_{epoch}.pt",
    )


def print_progress(epoch, n_epochs, len_dataloader, d_loss, g_loss):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{time}\t[{epoch}/{n_epochs}][{len_dataloader}/{len_dataloader}]\tLoss_D: {d_loss:.4f} Loss_G: {g_loss:.4f}"
    )


def print_progress(epoch, n_epochs, len_dataloader, d_loss, g_loss, d_acc):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{time}\t[{epoch}/{n_epochs}][{len_dataloader}/{len_dataloader}]\tLoss_D: {d_loss:.4f} Loss_G: {g_loss:.4f} Acc_D: {d_acc:.4f}"
    )


def train_gan(
    generator,
    discriminator,
    dataloader,
    optimizer_G,
    optimizer_D,
    adversarial_loss,
    device,
    args,
    history,
    sample_interval,
):
    g_losses = history["g_losses"]
    d_losses = history["d_losses"]

    fixed_noise = torch.randn(args.img_size, args.latent_size, 1, 1, device=device)

    os.makedirs(os.path.join(args.model_path, "images"), exist_ok=True)

    for epoch in range(args.current_epoch + 1, args.n_epochs + 1):
        g_loss_accum = 0.0
        d_loss_accum = 0.0

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
            g_loss_accum += g_loss.item()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            d_loss_accum += d_loss.item()

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(
                    gen_imgs.data[:25],
                    f"{args.model_path}/images/{batches_done}.png",
                    nrow=5,
                    normalize=True,
                )

        g_losses.append(g_loss_accum / len(dataloader))
        d_losses.append(d_loss_accum / len(dataloader))

        current_time = datetime.now()
        print(
            f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} [Epoch {epoch}/{args.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
        )

        if epoch % sample_interval == 0:
            history = {"g_losses": g_losses, "d_losses": d_losses}
            save_state(generator, discriminator, epoch, fixed_noise, args, history)


def train_dcgan_bce(
    generator,
    discriminator,
    dataloader,
    optimizerG,
    optimizerD,
    criterion,
    device,
    args,
    history,
    sample_interval,
):
    g_losses = history["g_losses"]
    d_losses = history["d_losses"]
    d_accs = history["d_accs"]

    fixed_noise = torch.randn(args.img_size, args.latent_size, 1, 1, device=device)

    real_label = 1.0
    fake_label = 0.0

    os.makedirs(os.path.join(args.model_path, "images"), exist_ok=True)

    for epoch in range(args.current_epoch + 1, args.n_epochs + 1):
        g_loss_accum = 0.0
        d_loss_accum = 0.0

        d_correct_real = 0
        d_correct_fake = 0
        total_real = 0
        total_fake = 0

        for i, data in enumerate(dataloader, 0):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            discriminator.zero_grad()

            # Train with real batch
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_data).view(-1)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Calulate discriminator real accuracy
            real_correct = (output > 0.5).float().eq(real_label).sum().item()
            total_real += real_data.size(0)
            d_correct_real += real_correct

            ## Train with fake batch
            noise = torch.randn(b_size, args.latent_size, 1, 1, device=device)

            fake_data = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_data.detach()).view(-1)

            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            # Calulate discriminator fake accuracy
            fake_correct = (output < 0.5).float().eq(fake_label).sum().item()
            total_fake += fake_data.size(0)
            d_correct_fake += fake_correct

            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            generator.zero_grad()

            label.fill_(real_label)
            output = discriminator(fake_data).view(-1)

            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()

            optimizerG.step()

            g_loss_accum += errG.item()
            d_loss_accum += errD.item()

        g_losses.append(g_loss_accum / len(dataloader))
        d_losses.append(d_loss_accum / len(dataloader))

        real_accuracy = d_correct_real / total_real
        fake_accuracy = d_correct_fake / total_fake
        combined_accuracy = (real_accuracy + fake_accuracy) / 2
        d_accs.append(combined_accuracy)

        print_progress(
            epoch,
            args.n_epochs,
            len(dataloader),
            d_losses[-1],
            g_losses[-1],
            d_accs[-1],
        )

        if epoch % sample_interval == 0 or epoch == args.n_epochs:
            history = {
                "g_losses": g_losses,
                "d_losses": d_losses,
                "d_accs": d_accs,
            }
            save_state(generator, discriminator, epoch, fixed_noise, args, history)


def train_dcgan_wasserstein(
    generator,
    discriminator,
    dataloader,
    optimizerG,
    optimizerD,
    device,
    args,
    history,
    sample_interval,
    clip_value=0.01,
):
    g_losses = history["g_losses"]
    d_losses = history["d_losses"]
    fixed_noise = torch.randn(args.img_size, args.latent_size, 1, 1, device=device)

    one = torch.tensor(1.0, device=device)
    mone = one * -1

    os.makedirs(os.path.join(args.model_path, "images"), exist_ok=True)

    for epoch in range(args.current_epoch + 1, args.n_epochs + 1):
        g_loss_accum = 0.0
        d_loss_accum = 0.0

        for i, data in enumerate(dataloader, 0):
            # (1) Update D network: maximize D(x) - D(G(z))
            discriminator.zero_grad()

            # Train with real batch
            real_data = data[0].to(device)
            D_real = discriminator(real_data).mean()
            D_real.backward(mone)

            # Train with fake batch
            noise = torch.randn(
                real_data.size(0), args.latent_size, 1, 1, device=device
            )
            fake_data = generator(noise)
            D_fake = discriminator(fake_data.detach()).mean()
            D_fake.backward(one)

            optimizerD.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # (2) Update G network: maximize D(G(z))
            generator.zero_grad()
            output = discriminator(fake_data).mean()
            output.backward(mone)
            optimizerG.step()

            g_loss_accum += -D_fake.item()
            d_loss_accum += D_real.item() - D_fake.item()

        g_losses.append(g_loss_accum / len(dataloader))
        d_losses.append(d_loss_accum / len(dataloader))

        print_progress(
            epoch, args.n_epochs, len(dataloader), d_losses[-1], g_losses[-1]
        )

        if epoch % sample_interval == 0 or epoch == args.n_epochs:
            history = {"g_losses": g_losses, "d_losses": d_losses}
            save_state(generator, discriminator, epoch, fixed_noise, args, history)
