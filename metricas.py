from util.util import desnormalize, load_model
from models.model import Model
from data import create_dataset
from options import Options_Test
import os, torch

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

# --------------------------------------------------------

path_root = os.path.abspath(__file__)
path_root = os.path.dirname(path_root)

opt = Options_Test()
dataset = create_dataset(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model(opt)

load_model(opt, model, device=device, train_as_valid=True)

model.require_grad(model.prior, False)
model.require_grad(model.encoder, False)
model.require_grad(model.generator, False)
model.require_grad(model.discriminator, False)

model.prior.eval()
model.encoder.eval()
model.generator.eval()

# --------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepararDados(dados):
    global device, opt

    return dados["img"].to(device), dados["pose"].to(device)

def to_numpy(img):
    img = img.permute(1, 2, 0)
    img = desnormalize(img)
    return img.numpy()

def save_loss(string):
    global opt

    with open(os.path.join(opt.path_raiz, "metricas.txt"), 'a') as f:
        f.write(string+'\n')

itr = 0

for dado in dataset:
    print("%d: " %itr, end='', flush=True)

    img, pose = prepararDados(dado)

    enc_mu, _ = model.encoder(img, pose)
    rec = model.generator(enc_mu, pose)

    img = img.cpu()
    rec = rec.cpu()

    for i in range(img.size(0)):
        original = to_numpy(img[i])
        fake = to_numpy(rec[i])

        psnr = PSNR(original, fake, data_range = (original.max() - original.min()))
        ssim = SSIM(original, fake, data_range = (original.max() - original.min()), multichannel=True)

        print(psnr, ssim, flush=True)

        string = "%f;%f" %(psnr, ssim)

        save_loss(string.replace('.', ','))

        #imshow(make_grid([img[i], rec[i]], 2))

    itr += 1

#1936