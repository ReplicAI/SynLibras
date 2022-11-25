from skimage.io import imsave
from skimage import img_as_ubyte
import zipfile, os, torch, torchvision
from .visualizer import imshow
import numpy as np
import random

def show_model(model, submodel=False):
    tparams = 0
    print('[Rede] %s\n' % type(model).__name__, flush=True) if not submodel else None

    for i, (name, child) in enumerate(model.named_children()):
        nparams = sum([p.numel() for p in child.parameters()])

        if submodel:
            print("\t%s: %3.3f M" % (name, (nparams / 1e6)), flush=True)
            continue

        print("%s: %3.3f M" % (name, (nparams / 1e6)), flush=True)

        tparams += nparams
        show_model(child, submodel=torch.triangular_solve)

    if submodel:
        return
            
    print('Número de parâmetros: %.3f M' % (tparams / 1e6), flush=True)
    print('\n' + '=' * 30, end='\n\n', flush=True)

def unzip(path_file, to_dir):            
    with zipfile.ZipFile(path_file) as zf:
        zf.extractall(to_dir)

    return True

def change(path1, path2, path_root):
    p1 = [v for v in path1.split("/") if v != ""]
    p2 = [v for v in path2.split("/") if v != ""]
    pr = [v for v in path_root.split("/") if v != ""]

    pr = pr[len(p1):]
    new_path = []
    new_path.extend(p2)
    new_path.extend(pr)

    new_path = os.path.join(*new_path)
    if not os.path.isabs(new_path) and len(path2) > 0 and path2[0] == '/':
        new_path = "/" + new_path

    return new_path

def save_loss(opt, iteration, losses):
    join_loss = {}
    for item in losses:
        for tipo in item:
            if not tipo in join_loss.keys():
                join_loss[tipo] = {}

            for loss in item[tipo]:
                if not loss in join_loss[tipo].keys():
                    join_loss[tipo][loss] = [item[tipo][loss].mean().item(), 1]
                else:
                    join_loss[tipo][loss][0] += item[tipo][loss].mean().item()
                    join_loss[tipo][loss][1] += 1

    result = "itr: %s, " %str(iteration)

    for tipo in join_loss:
        result += "%s: [" %tipo

        for loss in join_loss[tipo]:
            result += "%s: %.2f, " %(loss, join_loss[tipo][loss][0] / join_loss[tipo][loss][1])

        result = result[:-1] if result[-1] == ' ' else result
        result += '], '

    result = result[:-2] if result[-2:] == ", " else result
    result += '\n'

    if not os.path.isdir(opt.path_vars):
        os.makedirs(opt.path_vars)

    with open(os.path.join(opt.path_vars, "losses.txt"), 'a') as f:
        f.write(result)

def desnormalize(img):
    out = img / 2 + 0.5
    return out.clamp_(0.0, 1.0) # img * std + med,

def save_img(matriz, path_to):
    if type(matriz) != np.ndarray:
        with torch.no_grad():
            img = matriz.clone()
            img = img.to("cpu")

            if img.min() < 0.0: # "des"normalizar
                img = desnormalize(img)

            npimg = img.detach().numpy()
            if len(npimg.shape) == 3:
                imsave(path_to, img_as_ubyte(np.transpose(npimg, (1, 2, 0))))
            else:
                imsave(path_to, img_as_ubyte(npimg))

            return

    imsave(path_to, img_as_ubyte(matriz))

def make_gauss(img2d, center=(32,32), amplitude=1.0, std=(1.0,1.0)):
    varX, varY = std
    cX, cY = center
    A = amplitude
    coe1 = torch.tensor(float(2*varX))
    coe2 = float(2*varY)

    def pixel_value(x, y):
        #res = torch.exp(torch.tensor(-1* (((x-cX)**2/float(2*varX)) + ((y-cY)**2/float(2*varY))))) * A
        res = torch.exp(-1* (((x-cX)**2/coe1) + ((y-cY)**2/coe2))) * A
        return res

    for i in range(img2d.shape[0]):
        for j in range(img2d.shape[1]):
            img2d[i,j] = pixel_value(j, i) #invertido para que funcione corretamente o eixo x e y

#converte um ponto x, y de uma dimenção de imagem para outra
def convert_point_to_dim(ponto, from_size, to_size):
    xx = int((ponto[0] / float(from_size[0])) * to_size[0])
    yy = int((ponto[1] / float(from_size[1])) * to_size[1])

    xx = xx if xx < to_size[0] else to_size[0] - 1
    yy = yy if yy < to_size[1] else to_size[1] - 1
    x = xx if xx >= 0 else 0
    y = yy if yy >= 0 else 0
    return x, y

def convert_to_dim(ponto, from_size, to_size):
    if from_size[0] == 1920:
        ponto = [ponto[0], ponto[1]]
        ponto[0] = (ponto[0] - 448).clamp(0, 1024)
        ponto[1] = (ponto[1] - 56).clamp(0, 1024)

        return convert_point_to_dim(ponto, (1024, 1024), to_size)
    elif from_size[0] == 1024:
        ponto = [ponto[0], ponto[1]]
        ponto[0] = (ponto[0] - 256).clamp(0, 512)

        return convert_point_to_dim(ponto, (512, 512), to_size)

    return convert_point_to_dim(ponto, from_size, to_size)

def show_pose(pose, save=None, show=False):
    assert len(pose.shape) == 3
    assert pose.shape[1] == pose.shape[2]

    tam_pose = 12
    tam_face = 8
    tam_hand = 5

    img = torch.zeros(12, pose.shape[1], pose.shape[2])

    def multiplicador():
        mult = random.gauss(0.6, 0.4)
        mult = mult if mult >= 0.0 else mult * -1
        return torch.tensor(mult).clamp_(0.2, 1.0)

    #pose
    for i in range(tam_pose):
        img[0] += pose[i].clone() * multiplicador()
        img[1] += pose[i].clone() * multiplicador()
        img[2] += pose[i].clone() * multiplicador()

    soma = tam_pose

    #face
    for i in range(soma, soma+tam_face):
        img[3] += pose[i].clone() * multiplicador()
        img[4] += pose[i].clone() * multiplicador()
        img[5] += pose[i].clone() * multiplicador()

    soma += tam_face

    #hand
    for i in range(soma, soma+tam_hand):
        img[6] += pose[i].clone() * multiplicador()
        img[7] += pose[i].clone() * multiplicador()
        img[8] += pose[i].clone() * multiplicador()

    soma += tam_hand

    #hand
    for i in range(soma, soma+tam_hand):
        img[9] += pose[i].clone() * multiplicador()
        img[10] += pose[i].clone() * multiplicador()
        img[11] += pose[i].clone() * multiplicador()

    img.clamp_(0.0, 1.0)

    if show:
        imshow(
            torchvision.utils.make_grid(
                [img[:3], img[3:6], img[9:], img[6:9]], 2
            )
        )
    if save is not None:
        save_img(
            torchvision.utils.make_grid(
                [img[:3], img[3:6], img[9:], img[6:9]], 2
            ), save
        )

    #return img[:3]
    return torchvision.utils.make_grid([img[:3], img[3:6], img[9:], img[6:9]], 2)


def save_model(opt, model, optimP, optimE, optimG, optimD, epoca):
    path = os.path.join(opt.path_vars, "checkpoint")
    path += ".pt" if opt.train else "_valid.pt"

    if os.path.isfile(path):
        os.rename(path, path + ".bkp")

    torch.save({
        'epoca': epoca,
        'prior': model.prior.state_dict(),
        'encoder': model.encoder.state_dict(),
        'generator': model.generator.state_dict(),
        'discriminator': model.discriminator.state_dict(),
        'optimizerP': optimP.state_dict(),
        'optimizerE': optimE.state_dict(),
        'optimizerG': optimG.state_dict(),
        'optimizerD': optimD.state_dict(),
    }, path)
    
    if os.path.isfile(path + ".bkp"):
        os.remove(path + ".bkp")

def load_model(opt, model, optimP=None, optimE=None, optimG=None, optimD=None, device='cpu', train_as_valid=False):
    path = os.path.join(opt.path_vars, "checkpoint")
    path += ".pt" if opt.train or train_as_valid else "_valid.pt"

    print(path)

    if not os.path.isfile(path):
        print("** Os pesos da rede não foram encontrados. Partindo do inicio **\n")
        return 0

    checkpoint = torch.load(path, map_location=torch.device(device))

    model.prior.load_state_dict(checkpoint['prior'])
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.generator.load_state_dict(checkpoint['generator'])
    model.discriminator.load_state_dict(checkpoint['discriminator'])

    if opt.train:
        optimP.load_state_dict(checkpoint['optimizerP'])
        optimE.load_state_dict(checkpoint['optimizerE'])
        optimG.load_state_dict(checkpoint['optimizerG'])
        optimD.load_state_dict(checkpoint['optimizerD'])

        model.prior.train()
        model.encoder.train()
        model.generator.train()
        model.discriminator.train()

    print("Pesos Carregados [%d]!\n" %checkpoint['epoca'])

    return checkpoint['epoca']