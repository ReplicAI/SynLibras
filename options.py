import argparse, os, math

#args = parser.parse_args()
class Base_Options:
    def argParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("net_name", help="Nome da rede")
        parser.add_argument("checkpoint_dir", help="checkpoint dir")
        parser.add_argument("dataroot", help="Diretório do banco de dados (contendo duas pastas: frames e keypoints)")
        parser.add_argument("--batch_size", default=6, type=int, help="Tamanho do lote")

        return parser

    def __init__(self):
        self.parser = self.argParser()

        self.channels_img = 3
        self.channels_pose = 30
        self.ch_gen_out = 2 #deve ser de 1 a log2(size)
        self.tam_original_pose = (260, 260)
        self.size_pose = 256
        self.size_pose_gen = 128
        self.latent_dims = 256
        self.size = 256
        self.layers_pose = 4

        self.lr = 1e-3
        self.beta1 = 0
        self.beta2 = 0.99
        self.R1_once_every = 16
        self.scale_L1 = 0.1 / float( 2**max( (int( math.log2(self.size) ) - 4), 0 ) )
        self.scale_kl = 1.0
        self.scale_R1 = self.R1_once_every // 4

        #validação
        self.ignore_defeito = False

        self.epocas = 60
        self.val_freq = 2
        self.print_freq = 100

        #banco de dados:
        self.tam_gauss_menor = {64: 2, 128: 10, 256: 20}[self.size_pose]
        self.tam_gauss_maior = {64: 3, 128: 18, 256: 35}[self.size_pose]
        self.num_workers = 4 if self.size == 256 else 8

class Options_Train(Base_Options):
    args = None

    def parseArgs(self):
        if self.__class__.args is not None: return
        
        self.parser.add_argument("dataroot_valid", help="Diretório do banco de dados (contendo duas pastas: frames e keypoints)")
        self.__class__.args = self.parser.parse_args()

    def __init__(self, valid=False):
        super().__init__()

        self.parseArgs()
        args = self.__class__.args

        self.path_raiz = args.checkpoint_dir
        self.path_vars = os.path.join(args.checkpoint_dir, args.net_name)
        self.dataroot = args.dataroot_valid if valid else args.dataroot
        self.batch_size = args.batch_size

        self.shuffle = not valid
        self.train = not valid

class Options_Test(Base_Options):
    args = None

    def parseArgs(self):
        if self.__class__.args is not None: return
        
        self.parser.add_argument("--out_dir", help="Diretório de saída", required=True)
        self.__class__.args = self.parser.parse_args()

    def __init__(self):
        super().__init__()

        self.parseArgs()
        args = self.__class__.args

        self.path_vars = os.path.join(args.checkpoint_dir, args.net_name)
        self.dataroot = args.dataroot
        self.batch_size = args.batch_size
        self.out_dir = args.out_dir

        self.shuffle = False
        self.train = False