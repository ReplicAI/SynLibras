from .networks import Prior, Encoder, Generator, Discriminator
from .losses import rec_loss, divKLPrior, gan_loss
import torch, os

class Model:
    def __init__(self, opt):
        self.opt = opt

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.prior = Prior(opt).to(self.device)
        self.encoder = Encoder(opt).to(self.device)
        self.generator = Generator(opt).to(self.device)
        self.discriminator = Discriminator(opt).to(self.device)

    def require_grad(self, modulo, status):
        for p in modulo.parameters():
            p.requires_grad_(status)

    def get_snapshot(self, img, pose):
        self.prior.eval()
        self.encoder.eval()
        self.generator.eval()

        with torch.no_grad():
            enc_mu, _ = self.encoder(img, pose)
            img_recon = self.generator(enc_mu, pose)

            p_mu, p_logvar = self.prior(pose)
            z = self.reparametrize(p_mu, p_logvar)
            img_random = self.generator(z, pose)

            enc_mu, _ = self.encoder(img, torch.flip(pose, [0]))
            img_swap = self.generator(enc_mu, torch.flip(pose, [0]))

        self.prior.train()
        self.encoder.train()
        self.generator.train()

        return [img_recon, img_random, img_swap]

    def reparametrize(self, mu, logvar):
        device = mu.device
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def VAE(self, img, pose):
        enc_mu, enc_logvar = self.encoder(img, pose)
        p_mu, p_logvar = self.prior(pose)

        z_latent = self.reparametrize(enc_mu, enc_logvar)

        img_recon = self.generator(z_latent, pose)

        loss_rec = rec_loss(img_recon, img)
        kl = divKLPrior(enc_mu, enc_logvar, p_mu, p_logvar)

        return loss_rec, kl, z_latent.cpu()

    def GAN_random(self, img, pose):
        with torch.no_grad():
            mu, logvar = self.encoder(img, torch.flip(pose, [0]))
            z_latent = self.reparametrize(mu, logvar)

            img_fake = self.generator(z_latent, torch.flip(pose, [0]))

        mu, logvar = self.prior(pose)
        z_latent = self.reparametrize(mu, logvar)

        img_grad = self.generator(z_latent, pose)

        return img_grad, img_fake

    def GAN_swap(self, img, pose):
        with torch.no_grad():
            mu, logvar = self.prior(pose)
            z_latent = self.reparametrize(mu, logvar)

            img_fake = self.generator(z_latent, pose)

        mu, logvar = self.encoder(img, torch.flip(pose, [0]))
        z_latent = self.reparametrize(mu, logvar)

        img_grad = self.generator(z_latent, torch.flip(pose, [0]))

        return img_grad, img_fake

    def Sep_GAN_discriminator(self, img, pose, swap=False):
        func = self.GAN_swap if swap else self.GAN_random

        img_grad, img_fake = func(img, pose)

        real = self.discriminator(img)
        fake_1 = self.discriminator(img_grad.detach())
        fake_2 = self.discriminator(img_fake)

        l_real = gan_loss(real, True)
        l_fake_1 = gan_loss(fake_1, False)
        l_fake_2 = gan_loss(fake_2, False)

        l_fake = (l_fake_1 + l_fake_2) * 0.5

        pred_real = real.mean().to('cpu')
        pred_fake = (fake_1.mean().to('cpu') + fake_2.mean().to('cpu')) * 0.5

        lossD = (l_real + l_fake).mean()

        return lossD, pred_real, pred_fake, img_grad

    def Sep_GAN_generator(self, img_grad):
        pred = self.discriminator(img_grad)

        lossG = gan_loss(pred, True)

        return lossG.mean()

    def GAN_discriminator(self, img, pose):
        #random
        mu, logvar = self.prior(pose)
        z_latent = self.reparametrize(mu, logvar)
        img_random = self.generator(z_latent, pose)

        #swap
        mu, logvar = self.encoder(img, torch.flip(pose, [0]))
        z_latent = self.reparametrize(mu, logvar)
        img_swap = self.generator(z_latent, torch.flip(pose, [0]))

        real = self.discriminator(img)
        fake_1 = self.discriminator(img_random.detach())
        fake_2 = self.discriminator(img_swap.detach())

        l_real = gan_loss(real, True)
        l_fake_1 = gan_loss(fake_1, False)
        l_fake_2 = gan_loss(fake_2, False)

        l_fake = (l_fake_1 + l_fake_2) * 0.5

        pred_real = real.mean().to('cpu')
        pred_fake = (fake_1.mean().to('cpu') + fake_2.mean().to('cpu')) * 0.5

        lossD = (l_real + l_fake).mean()

        return lossD, pred_real, pred_fake, img_random, img_swap

    def GAN_generator(self, img_random, img_swap):
        pred_1 = self.discriminator(img_random)
        pred_2 = self.discriminator(img_swap)

        lossG_1 = gan_loss(pred_1, True)
        lossG_2 = gan_loss(pred_2, True)

        lossG = (lossG_1 + lossG_2).mean()

        return lossG

    def R1_loss(self, img):
        img.requires_grad_()
        pred_real = self.discriminator(img).sum()

        grad_real, = torch.autograd.grad(
            outputs=pred_real,
            inputs=[img],
            create_graph=True,
            retain_graph=True,
        )

        grad_real2 = grad_real.pow(2)
        dims = list(range(1, grad_real2.ndim))
        grad_penalty = grad_real2.sum(dims) * (0.5 * self.opt.scale_R1)
        grad_penalty = grad_penalty.mean()

        return grad_penalty