from util import imshow, save_img
from util.util import load_model
from models.model import Model
import torchvision.transforms as transforms
from options import Options_Test
import os, torch, json

from skimage.io import imread

from torchvision.utils import make_grid
from data.pose import PosesGenerator

path_root = os.path.abspath(__file__)
path_root = os.path.dirname(path_root)

path_dados = "/home/wellington/Documentos/CP e dataset/Dados Utilizados/Dados Utilizados no Artigo/transferencias/CP/videogame"
path_to = "/home/wellington/Documentos/CP e dataset/resultados/transferencias CP/videogame"

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
    global path_dados

    ator = imread(os.path.join(path_dados, filename))
    ator = torch.from_numpy(ator).permute(2, 0, 1) / 255.0

    if ator.shape[-1] != ator.shape[-2]:
        x = torch.zeros(3, 260, 260, dtype=ator.dtype)
        x[:, :, 25:-25] = ator
        return x

    return ator

ator = loadImg("47.png")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ator = transform(ator.unsqueeze(0)).repeat(6, 1, 1, 1)

imshow(make_grid(ator, 6))


pose_est = PosesGenerator(
    (opt.size, opt.size),
    tam_gauss=[opt.tam_gauss_menor, opt.tam_gauss_maior],
    tam_original_pose=(1920, 1080)
)

listas_poses = [
    os.path.join(path_dados, "1_keypoints.json"),
    os.path.join(path_dados, "29_keypoints.json"),
    os.path.join(path_dados, "36_keypoints.json"),
    os.path.join(path_dados, "64_keypoints.json"),
    os.path.join(path_dados, "116_keypoints.json"),
    os.path.join(path_dados, "155_keypoints.json"),
]

all_poses = torch.zeros(6, opt.channels_pose, opt.size, opt.size)

for i, linkpose in enumerate(listas_poses):
    with open(linkpose, "r") as f:
        data = json.load(f)
        p = pose_est.make(data)

        all_poses[i] = p

imshow(make_grid(
    list(map(lambda x: x[9:12], all_poses)), 6
))

ator = ator.float().to(device)
all_poses = all_poses.float().to(device)

print(type(ator), type(all_poses))

enc_mu, _ = model.encoder(ator, all_poses)
rec = model.generator(enc_mu, all_poses)

save_img(make_grid(rec, rec.size(0)), os.path.join(path_to, "transferencia.png"))