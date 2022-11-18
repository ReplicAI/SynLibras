import torchvision.transforms as transforms
import os, torch, json

from util import imshow, save_img
from util.util import load_model
from models.model import Model
from options import Options_Test

from skimage.io import imread

from torchvision.utils import make_grid
from data.pose import PosesGenerator

path_root = os.path.abspath(__file__)
path_root = os.path.dirname(path_root)

path_atores = "/home/wellington/Documentos/CP testes/pontos_transferencia/seleção pra transferencia/atores"
path_jsons = "/home/wellington/Documentos/CP testes/keypoints_agua"

# -------------------------------------------------------- Carregando modelo

opt = Options_Test()
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

# -------------------------------------------------------- Carregando dados

def loadImg(filename):
    global path_atores

    ator = imread(os.path.join(path_atores, filename))
    ator = torch.from_numpy(ator).permute(2, 0, 1) / 255.0

    if ator.shape[-1] != ator.shape[-2]:
        x = torch.zeros(3, 260, 260, dtype=ator.dtype)
        x[:, :, 25:-25] = ator
        return x

    return ator


ator0 = loadImg("2.png")

#print(ator0.min(), ator0.max(), ator0.shape)

ator1 = loadImg("0.png")

#print(ator1.min(), ator1.max(), ator1.shape)

ator2 = loadImg("01April_2011_Friday_tagesschau.avi_pid0_fn000000-0.png")

#print(ator2.min(), ator2.max(), ator2.shape)

ator3 = loadImg("04January_2010_Monday_tagesschau.avi_pid0_fn000093-0.png")

#print(ator3.min(), ator3.max(), ator3.shape)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ator0 = transform(ator0.unsqueeze(0))
ator1 = transform(ator1.unsqueeze(0))
ator2 = transform(ator2.unsqueeze(0))
ator3 = transform(ator3.unsqueeze(0))


imshow(make_grid(
    [
        make_grid(ator0, 1),
        make_grid(ator1, 1),
        make_grid(ator2, 1),
        make_grid(ator3, 1)
    ], 1
))

pose_est = PosesGenerator(
    (opt.size, opt.size),
    tam_gauss=[opt.tam_gauss_menor, opt.tam_gauss_maior],
    tam_original_pose=(1920, 1080)
)

path_pose = os.path.join(path_jsons, "30_keypoints.json")

pose = torch.zeros(1, opt.channels_pose, opt.size, opt.size)

with open(path_pose, "r") as f:
    data = json.load(f)
    pose[0] = pose_est.make(data)


#interpolação

ator0 = ator0.float().to(device)
ator1 = ator1.float().to(device)
ator2 = ator2.float().to(device)
ator3 = ator3.float().to(device)
pose = pose.float().to(device)


enc_mu0, _ = model.encoder(ator0, pose)
enc_mu1, _ = model.encoder(ator1, pose)
enc_mu2, _ = model.encoder(ator2, pose)
enc_mu3, _ = model.encoder(ator3, pose)

qtd = 11

passo = 1 / (qtd - 1)
lista_rec = []

for i in range(qtd):
    lista_rec.append([])

    mp = i * passo
    inter_mu0 = ( (1 - mp) * enc_mu0 ) + ( mp * enc_mu2 )
    inter_mu1 = ( (1 - mp) * enc_mu1 ) + ( mp * enc_mu3 )

    for j in range(qtd):
        mp = j * passo
        latent = ( (1 - mp) * inter_mu0 ) + ( mp * inter_mu1 )

        lista_rec[-1].append(model.generator(latent, pose)[0])

save_img(make_grid([
    make_grid(lista, len(lista)) for lista in lista_rec
], 1), os.path.join(path_root, "interpolation.png"))

#save_img(make_grid(rec0, rec0.size(0)), os.path.join(path_root, "0.png"))
#save_img(make_grid(rec1, rec1.size(0)), os.path.join(path_root, "1.png"))
#save_img(make_grid(rec2, rec2.size(0)), os.path.join(path_root, "2.png"))

exit()

imshow(
    make_grid(
        make_grid(rec0, rec0.size(0)),
        make_grid(rec1, rec1.size(0)),
        make_grid(rec2, rec2.size(0))
    , 1)
)
