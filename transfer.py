from util import save_img, desnormalize
from data.pose import PosesGenerator
from util.util import load_model
from options import Options_Test

from models.model import Model

from skimage.io import imread

import torch, os, json
import torchvision.transforms as transforms

opt = Options_Test()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isdir(opt.out_dir):
    os.makedirs(opt.out_dir)

frames = os.path.join(opt.dataroot, "frames")
keypoints = os.path.join(opt.dataroot, "keypoints")

if not os.path.isdir(frames):
    raise FileNotFoundError(frames)

if not os.path.isdir(keypoints):
    raise FileNotFoundError(keypoints)

# ler dados do disco

images_path = []
poses_path = []

for name in os.listdir(frames):
    images_path.append(os.path.join(frames, name))

for name in sorted(os.listdir(keypoints)):
    poses_path.append(os.path.join(keypoints, name))

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

# transform

transform = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

pose_est = PosesGenerator(
    (opt.size, opt.size),
    tam_gauss=[opt.tam_gauss_menor, opt.tam_gauss_maior],
    tam_original_pose=opt.tam_original_pose
)

# loop

for img_path in images_path:
    img = imread(img_path)
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0

    for i in range(0, len(poses_path), opt.batch_size):
        sub_poses_path = poses_path[i:i+opt.batch_size]

        all_poses = torch.zeros(len(sub_poses_path), opt.channels_pose, opt.size, opt.size)

        for itr, path_pose in enumerate(sub_poses_path):
            with open(path_pose, "r") as f:
                data = json.load(f)
                p = pose_est.make(data)

                all_poses[itr] = p

        # make prediction

        ator = transform(img.unsqueeze(0)).repeat(len(sub_poses_path), 1, 1, 1)

        # preprocess

        ator = ator.float().to(device)
        all_poses = all_poses.float().to(device)        

        enc_mu, _ = model.encoder(ator, all_poses)
        img_recon = model.generator(enc_mu, all_poses)

        for itr, path_pose in enumerate(sub_poses_path):
            imgname = '.'.join(os.path.basename(img_path).split('.')[:-1])
            posename = '.'.join(os.path.basename(path_pose).split('.')[:-1])

            posename = posename.replace("_keypoints", '')

            path_to = os.path.join(opt.out_dir, imgname, f"{imgname}_{posename}.png")

            if not os.path.isdir(os.path.dirname(path_to)):
                os.makedirs(os.path.dirname(path_to))

            save_img(desnormalize(img_recon[itr].cpu()),path_to)
