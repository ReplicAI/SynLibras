from util.util import make_gauss, convert_to_dim
from skimage.transform import rotate, resize
import torch, math

#essa classe gera uma lista de camadas de gaussianas de forma mais rapida
class PosesGenerator:
    def __init__(self, size_img=(), tam_gauss=[32, 64], tam_original_pose=(260, 260)):
        assert len(size_img) == 2 and len(tam_gauss) == 2
        self.tamanho_gauss = tam_gauss
        self.size_img = size_img
        self.tam_gauss = list(tam_gauss)
        self.c = 0.005
        self.cc = 1e-12
        
        self.tam_original_pose = tam_original_pose

        self.infos_to_make = {
            "pose": {
                "corpo": (0, 1, 8), "right": (2, 3, 4), "left": (5, 6, 7)},
            "face": [40, 41, 46, 47, 62, 66, 68, 69],
            "hand": [4, 8, 12, 16, 20]
        }

        num_to_par = 10 if tam_gauss[0] % 2 == 0 else 11
        self.tam_gauss[0] += num_to_par

        #gerar uma gaussiana menor
        self.gauss = torch.zeros(self.tam_gauss[0], self.tam_gauss[0])
        make_gauss(
            self.gauss,
            center=(self.tam_gauss[0]//2,self.tam_gauss[0]//2),
            amplitude=1,
            std=(
                self.tam_gauss[0]-num_to_par,
                self.tam_gauss[0]-num_to_par
            )
        )

        num_to_par = 6 if tam_gauss[1] % 2 == 0 else 7
        self.tam_gauss[1] += num_to_par

        #gerar uma gaussiana maior
        self.gauss_maior = torch.zeros(self.tam_gauss[1], self.tam_gauss[1])

        make_gauss(
            self.gauss_maior,
            center=(self.tam_gauss[1]//2, self.tam_gauss[1]//2),
            amplitude=1,
            std=(self.tam_gauss[1]-num_to_par, self.tam_gauss[1]-num_to_par)
        )

        self.tam_mask = 48 if self.size_img == 64 else 160

        #gaussiana para gerar os membros
        self.gauss_espichado = torch.zeros(self.tam_mask, self.tam_mask)
        make_gauss(
            self.gauss_espichado, center=(self.tam_mask//2, self.tam_mask//2), amplitude=1,
            std=(650 if self.tam_mask == 160 else 64, self.tamanho_gauss[1])
        )

    def __make_gauss_list(self, list_center, maior=False):
        assert len(list_center) > 0

        itr = 1 if maior else 0

        #criar imagem de saida
        img_full = torch.zeros(self.size_img[0]+self.tam_gauss[itr], self.size_img[1]+self.tam_gauss[itr])

        #rodar todos os pontos que deve ter uma gaussiana
        for ponto in list_center:
            x, y = ponto

            #como img_full é maior que a imagem de saida, temos que fixar o ponto 0,0 da imagem menor com
            #a compensação da imagem maior
            center = (x + self.tam_gauss[itr]//2, y + self.tam_gauss[itr]//2)

            xi, xf = center[0] - self.tam_gauss[itr]//2, center[0] + self.tam_gauss[itr]//2
            yi, yf = center[1] - self.tam_gauss[itr]//2, center[1] + self.tam_gauss[itr]//2

            img_full[xi:xf, yi:yf] += self.gauss_maior if maior else self.gauss

        return torch.transpose(
            img_full[self.tam_gauss[itr]//2:-self.tam_gauss[itr]//2, self.tam_gauss[itr]//2:-self.tam_gauss[itr]//2],
                1, 0)

    def get_preenchimento(self, ponto1, ponto2):
        ponto1 = ponto1.tolist() if type(ponto1) == torch.Tensor else ponto1
        ponto2 = ponto2.tolist() if type(ponto2) == torch.Tensor else ponto2

        def dist(p1=(), p2=()):
            return math.sqrt( (p2[0]-p1[0])**2 + (p2[1] - p1[1])**2 )

        out = torch.zeros(self.size_img[0]+self.tam_mask, self.size_img[1]+self.tam_mask)

        std_x = int(dist((ponto1[0], ponto1[1]), (ponto2[0], ponto2[1])) * 1.8)
        std_x = std_x if std_x % 2 == 0 else std_x + 1
        std_x = self.tam_mask if std_x > self.tam_mask else std_x
        std_x = 2 if std_x < 2 else std_x

        cx = (abs(ponto1[0]+ponto2[0]) // 2) + (self.tam_mask // 2)
        cy = (abs(ponto1[1]+ponto2[1]) // 2) + (self.tam_mask // 2)

        ang = math.atan(
            (ponto2[1]-ponto1[1]) / (ponto2[0]-ponto1[0])
        ) * 180/math.pi if (ponto2[0]-ponto1[0]) != 0 else 90

        cx = cx.item() if type(cx) == torch.Tensor else cx
        cy = cy.item() if type(cy) == torch.Tensor else cy

        es = torch.from_numpy(resize(self.gauss_espichado.numpy(), (self.tam_mask, std_x)))

        out[cy-(self.tam_mask // 2) : cy+(self.tam_mask // 2), cx - es.size(1)//2 : cx + es.size(1)//2] = es

        out = out[(self.tam_mask // 2):-(self.tam_mask // 2), (self.tam_mask // 2):-(self.tam_mask // 2)]

        return torch.from_numpy(rotate(out.numpy(), ang*-1, center=(cx-(self.tam_mask // 2), cy-(self.tam_mask // 2))))

    #essa função auxilia na função make_pose
    #dado uma lista de pontos, gera uma saida de pontos gaussiano representando a pose
    def __make_matriz_from_points(self, lista_pontos, tipo, tam_orig_pose):
        tam_orig_pose = self.tam_original_pose if tam_orig_pose is None else tam_orig_pose
        local_pose = torch.zeros(1, *self.size_img)
        all_pontos = torch.tensor(lista_pontos).view(-1, 3)

        if tipo == "pose":
            for key in self.infos_to_make["pose"]:
                i, j, k = self.infos_to_make["pose"][key]

                if all_pontos[i][2] >= self.c:
                    p1 = convert_to_dim(all_pontos[i][:2], tam_orig_pose, self.size_img)
                    local_pose[-1] = self.__make_gauss_list([p1], maior=False)
                local_pose = torch.cat([local_pose, local_pose[-1:]], dim=0)
                
                if all_pontos[j][2] >= self.c:
                    p2 = convert_to_dim(all_pontos[j][:2], tam_orig_pose, self.size_img)
                    local_pose[-1] = self.__make_gauss_list([p2], maior=False)
                local_pose = torch.cat([local_pose, local_pose[-1:]], dim=0)
                
                if all_pontos[k][2] >= (self.c if k != 8 else self.cc):
                    p3 = convert_to_dim(all_pontos[k][:2], tam_orig_pose, self.size_img)
                    local_pose[-1] = self.__make_gauss_list([p3], maior=False)
                local_pose = torch.cat([local_pose, local_pose[-1:]], dim=0)

            for key in self.infos_to_make["pose"]:
                i, j, k = self.infos_to_make["pose"][key]

                p1 = convert_to_dim(all_pontos[i][:2], tam_orig_pose, self.size_img)
                p2 = convert_to_dim(all_pontos[j][:2], tam_orig_pose, self.size_img)
                p3 = convert_to_dim(all_pontos[k][:2], tam_orig_pose, self.size_img)

                if all_pontos[j][2] >= self.c:
                    if all_pontos[i][2] >= self.c and all_pontos[k][2] >= (self.c if k != 8 else self.cc):
                        local_pose[-1] = self.get_preenchimento(p1, p2)
                        local_pose[-1] += self.get_preenchimento(p2, p3)
                    elif all_pontos[i][2] >= self.c:
                        local_pose[-1] = self.get_preenchimento(p1, p2)
                    else:
                        local_pose[-1] = self.get_preenchimento(p2, p3)
                else:
                    local_pose[-1].fill_(0)
                    
                local_pose = torch.cat([local_pose, local_pose[-1:]], dim=0)

            return local_pose[:-1]

        for p in self.infos_to_make[tipo]:
            if all_pontos[p][2] < self.c:
                local_pose[-1].fill_(0)
                local_pose = torch.cat([local_pose, local_pose[-1:]], dim=0)
                continue

            local_pose[-1] = self.__make_gauss_list([
                convert_to_dim(
                    (all_pontos[p][0], all_pontos[p][1]), tam_orig_pose, self.size_img
            )], maior=False)

            local_pose = torch.cat([local_pose, local_pose[-1:]], dim=0)

        return local_pose[:-1]

    def make(self, infos_json, tam_original_pose=None):
        pose = torch.zeros(1, *self.size_img)
        face = torch.zeros(1, *self.size_img)
        hand = torch.zeros(1, *self.size_img)
        
        for k in infos_json['people'][0]:
            if k == 'person_id' or '3d' in k:
                continue

            if "pose" in k:
                m = self.__make_matriz_from_points(infos_json['people'][0][k], "pose", tam_original_pose)
                pose = torch.cat([pose, m], dim=0)
            elif "face" in k:
                m = self.__make_matriz_from_points(infos_json['people'][0][k], "face", tam_original_pose)
                face = torch.cat([face, m], dim=0)
            elif "hand" in k:
                m = self.__make_matriz_from_points(infos_json['people'][0][k], "hand", tam_original_pose)
                hand = torch.cat([hand, m], dim=0)

        label = torch.cat([pose[1:], face[1:], hand[1:]], dim=0)

        return label.clamp_(0.0, 1.0)