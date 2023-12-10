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
        "seed": args.seed,
        "history": history,
    }
    with open(f"{args.model_path}/parameters_dict.json", "w") as file:
        json.dump(current_params, file)

    torch.save(generator.state_dict(), f"{args.model_path}/generator_epoch_{epoch}.pt")
    torch.save(
        discriminator.state_dict(),
        f"{args.model_path}/discriminator_epoch_{epoch}.pt",
    )


def print_progress(epoch, n_epochs, batch, n_batch, d_loss, g_loss, d_acc=None):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_msg = f"{time}\t[{epoch}/{n_epochs}][{batch}/{n_batch}]\tLoss_D: {d_loss:.4f} Loss_G: {g_loss:.4f}"
    if d_acc is not None:
        progress_msg += f" Acc_D: {d_acc:.4f}"
    print(progress_msg)


def train_gan_bce(
    generator,
    discriminator,
    dataloader,
    optimizerG,
    optimizerD,
    criterion,
    device,
    args,
    history,
    fixed_noise,
    sample_interval,
):
    torch.use_deterministic_algorithms(False)

    g_losses = history["g_losses"]
    d_losses = history["d_losses"]
    d_accs = history["d_accs"]

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
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            discriminator.zero_grad()

            # Train with real batch
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_data).view(-1)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Calculate discriminator real accuracy
            real_correct = (output > 0.5).float().sum().item()
            total_real += b_size
            d_correct_real += real_correct

            # Train with fake batch
            noise = torch.randn(b_size, args.latent_size, device=device)
            fake_data = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_data.detach()).view(-1)

            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            # Calculate discriminator fake accuracy
            fake_correct = (output < 0.5).float().sum().item()
            total_fake += b_size
            d_correct_fake += fake_correct

            optimizerD.step()

            # Update G network: maximize log(D(G(z)))
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
    fixed_noise,
    sample_interval,
):
    g_losses = history["g_losses"]
    d_losses = history["d_losses"]
    d_accs = history["d_accs"]

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
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
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

            # Train with fake batch
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

            # Update G network: maximize log(D(G(z)))
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
    fixed_noise,
    sample_interval,
    clip_value=0.01,
):
    g_losses = history["g_losses"]
    d_losses = history["d_losses"]

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
            epoch,
            args.n_epochs,
            len(dataloader),
            len(dataloader),
            d_losses[-1],
            g_losses[-1],
        )

        if epoch % sample_interval == 0 or epoch == args.n_epochs:
            history = {"g_losses": g_losses, "d_losses": d_losses}
            save_state(generator, discriminator, epoch, fixed_noise, args, history)
