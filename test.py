from util import save_img, desnormalize
from util.util import load_model
from options import Options_Test
from data import create_dataset
import torch, os

from models.model import Model

opt = Options_Test()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isdir(opt.out_dir):
    os.makedirs(opt.out_dir)

#-- dados
dataset = create_dataset(opt)

#-- modelo
model = Model(opt)

load_model(opt, model, device=device)

model.require_grad(model.prior, False)
model.require_grad(model.encoder, False)
model.require_grad(model.generator, False)
model.require_grad(model.discriminator, False)

model.prior.eval()
model.encoder.eval()
model.generator.eval()

def prepararDados(dados):
    global device, opt

    return dados["img"].to(device), dados["pose"].to(device)

#reconstrução
for data in dataset:
    img, pose = prepararDados(data)

    enc_mu, _ = model.encoder(img, pose)
    img_recon = model.generator(enc_mu, pose)

    for i in range(img_recon.size(0)):
        name = data["name"][i]

        save_img(
            desnormalize(img_recon[i].cpu()),
            os.path.join(opt.out_dir, name)
        )