#python3 train.py rede_teste "/home/wellington/Documentos/CP testes" /home/wellington/Downloads/treinamento_phoenix_19365 -bs 2
#python3 train.py rede_teste "/home/wellington/Downloads/rede_teste-20220913T161044Z-001" /home/wellington/Downloads/treinamento_phoenix_19365 -bs 2

from util import TensorBoard
from util.util import save_model, load_model
from options import Options_Train
from data import create_dataset
import torch

from models.model import Model

# -------------------------------------------------------- Definições

opt = Options_Train()
opt_valid = Options_Train(valid=True)

#dataset treinamento
dataset = create_dataset(opt)

#dataset validação
dataset_valid = create_dataset(opt_valid)

#obter lote de comparação
lote_comp = next(iter(dataset_valid))

tb = TensorBoard(opt)

model = Model(opt)

# -------------------------------------------------------- Otimizadores

optimizerP = torch.optim.Adam(
    model.prior.parameters(),
    lr = opt.lr, betas=(opt.beta1, opt.beta2)
)

optimizerE = torch.optim.Adam(
    model.encoder.parameters(),
    lr = opt.lr, betas=(opt.beta1, opt.beta2)
)

optimizerG = torch.optim.Adam(
    model.generator.parameters(),
    lr = opt.lr, betas=(opt.beta1, opt.beta2)
)

c = opt.R1_once_every / (1 + opt.R1_once_every)

optimizerD = torch.optim.Adam(
    model.discriminator.parameters(),
    lr = opt.lr * c, betas=(opt.beta1**c, opt.beta2**c)
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------- Preparar dados a cada iteração

def prepararDados(dados):
    global device, opt

    return dados["img"].to(device), dados["pose"].to(device)

# -------------------------------------------------------- Carregar modelo

start = load_model(opt, model, optimizerP, optimizerE, optimizerG, optimizerD, device)

print("Inicio do Treinamento")

for epoch in range(start + 1, opt.epocas + 1):
    ### ----------------------- Antes de treinar -----------------------------

    with torch.no_grad():
        img, pose = prepararDados(lote_comp)
        result = model.get_snapshot(img, pose)

        tb.add_img_comp(img, result, epoch-1)

    print(f'epoca {epoch:3d} [ ', end='')

    vaeRec_loss = vaeKL_loss = ganG_loss = ganD_loss = 0.0

    ### ----------------------- Loop de treinamento -----------------------------

    for i, data in enumerate(dataset, 0):
        img, pose = prepararDados(data)

        ### ----------------------- VAE -----------------------------

        model.require_grad(model.prior, True)
        model.require_grad(model.encoder, True)
        model.require_grad(model.generator, True)
        model.require_grad(model.discriminator, False)

        loss_rec, loss_kl, z_latent = model.VAE(img, pose)

        lossVAE = (loss_rec * opt.scale_L1) + (loss_kl * opt.scale_kl)

        optimizerP.zero_grad()
        optimizerE.zero_grad()
        optimizerG.zero_grad()

        lossVAE.backward()

        optimizerP.step()
        optimizerE.step()
        optimizerG.step()

        ### ----------------------- GAN -----------------------------

        model.require_grad(model.prior, False)
        model.require_grad(model.encoder, False)
        model.require_grad(model.generator, True)
        model.require_grad(model.discriminator, True)

        #img, pose = prepararDados(next(data_disc_iter))

        lossD, pred_real, pred_fake, img_random, img_swap = model.GAN_discriminator(img, pose)

        optimizerD.zero_grad()

        lossD.backward()

        optimizerD.step()

        #generator
        lossG = model.GAN_generator(img_random, img_swap)

        optimizerG.zero_grad()

        lossG.backward()

        optimizerG.step()

        ### ----------------------- Salvar erros -----------------------------

        vaeRec_loss += loss_rec.item()
        vaeKL_loss += loss_kl.item()
        ganG_loss += lossG.item()
        ganD_loss += lossD.item()

        ### ----------------------- R1 Loss -----------------------------

        if i % opt.R1_once_every == 0 and i > 0:
            grad_penalty = model.R1_loss(img)

            optimizerD.zero_grad()

            grad_penalty.backward()

            optimizerD.step()

        ### ----------------------- Outras operações -----------------------------

        if i == (epoch - 1) == 0:
            print(f"\nlosses: rec {vaeRec_loss:.2f} kl {vaeKL_loss:.2f} gen {ganG_loss:.2f} disc {ganD_loss:.2f} pred_real {pred_real:.2f} pred_fake {pred_fake:.2f}\n[ ", end='')

        if i % opt.print_freq == 0:
            print('=', end='', flush=True)

    ### ----------------------- Processado por epoca -----------------------------

    save_model(opt, model, optimizerP, optimizerE, optimizerG, optimizerD, epoch)

    vaeRec_loss /= i
    vaeKL_loss /= i
    ganG_loss /= i
    ganD_loss /= i

    print(f" ] losses: rec {vaeRec_loss:.2f} kl {vaeKL_loss:.2f} gen {ganG_loss:.2f} disc {ganD_loss:.2f} pred_real {pred_real:.2f} pred_fake {pred_fake:.2f}", end='')

    losses = {
        "VAE": { "rec": vaeRec_loss, "kl": vaeKL_loss },
        "loss_gen": { "loss_G": ganG_loss },
        "loss_disc": { "lossD": ganD_loss},
        "pred": { "real": pred_real, "fake": pred_fake }
    }

    tb.add_loss_train(losses, epoch)

    ### ----------------------- Validação -----------------------------

    if (epoch - 1) % opt.val_freq == 0:
        print(" val")

        with torch.no_grad():
            vaeRec_loss = vaeKL_loss = 0.0

            for i, data in enumerate(dataset_valid, 0):
                img, pose = prepararDados(data)

                loss_rec, loss_kl, z_latent = model.VAE(img, pose)

                vaeRec_loss += loss_rec.item()
                vaeKL_loss += loss_kl.item()

            vaeRec_loss /= i
            vaeKL_loss /= i

            losses = {
                "VAE": { "rec": vaeRec_loss, "kl": vaeKL_loss },
            }

            tb.add_loss_valid(losses, epoch)

            if tb.needs_save_valid_model():
                tb.checkpoint(epoch)
                save_model(opt_valid, model, optimizerP, optimizerE, optimizerG, optimizerD, epoch)

            tb.done()
            tb.reload()
    else:
        print(" ")

print("Fim do Treinamento")