import torchvision.transforms as transforms
import os, json, pickle, io
import torch, torchvision
import warnings

from data.pose import PosesGenerator
from PIL import Image

class Phoenix:
    def __init__(self, opt):
        self.frames = os.path.join(opt.dataroot, "frames")
        self.keypoints = os.path.join(opt.dataroot, "keypoints")

        #verificar a existência dos diretórios
        if not os.path.isdir(self.frames):
            raise FileNotFoundError(self.frames)

        if not os.path.isdir(self.keypoints):
            raise FileNotFoundError(self.keypoints)

        #dados
        self.data = os.listdir(self.frames)

        if len(self.data) != len(os.listdir(self.keypoints)):
            if len(self.data) > len(os.listdir(self.keypoints)):
                raise Exception("the frames path has more images than the keypoints path has keypoints")

            warnings.warn("frames path and keypoints path have a different file count", UserWarning)

        self.transform = transforms.Compose([
            transforms.Resize((opt.size, opt.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.gerador_de_poses = PosesGenerator(
           (opt.size_pose, opt.size_pose),
           (opt.tam_gauss_menor, opt.tam_gauss_maior),
           tam_original_pose=opt.tam_original_pose
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_img = os.path.join(self.frames, self.data[idx])

        namepose = '.'.join(self.data[idx].split('.')[:-1]) + "_keypoints.json"
        path_pose = os.path.join(self.keypoints, namepose)

        if not os.path.isfile(path_img):
            raise FileNotFoundError("frame:", path_img)

        if not os.path.isfile(path_pose):
            raise FileNotFoundError("keypoint:", path_pose)

        #carregar

        img = Image.open(path_img).convert('RGB')

        with open(path_pose, 'rb') as f:
            pose = self.gerador_de_poses.make(json.load(f))
        
        img = self.transform(img)
        
        return {"img": img, "pose": pose, "name": self.data[idx]}