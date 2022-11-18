from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from .util import desnormalize, save_img
import os, torch
import pickle

class TensorBoard:
    def __init__(self, opt):
        self.opt = opt

        self.path_files = os.path.join(opt.path_vars, "runs")
        self.path_losses_valid = os.path.join(opt.path_vars, "loss.valid")
        self.losses_valid = []
        self.monitorar = "rec"

        if not os.path.isdir(opt.path_vars):
            os.makedirs(opt.path_vars)

        if os.path.isfile(self.path_losses_valid):
            with open(self.path_losses_valid, 'rb') as f:
                self.losses_valid = pickle.load(f)
        
        self.path_snaps = os.path.join(opt.path_raiz, "snapshots")

        self.reload()

    def reload(self):
        self.tb = SummaryWriter(self.path_files)

    def done(self):
        self.tb.flush()
        self.tb.close()

        with open(self.path_losses_valid, 'wb') as f:
            pickle.dump(self.losses_valid, f)

    def needs_save_valid_model(self):
        if len(self.losses_valid) <= 1:
            return False

        if self.losses_valid[-1] < min(self.losses_valid[:-1]):
            return True

        return False

    def checkpoint(self, epoca):
        self.tb.add_scalar('checkpoint', 1.0, epoca)

    def add_loss_valid(self, losses, epoca):
        for tipo in losses:
            for loss in losses[tipo]:

                if loss == self.monitorar:
                    self.losses_valid.append(losses[tipo][loss])

                self.tb.add_scalar('validation ' + loss, losses[tipo][loss], epoca)

    def add_loss_train(self, losses, epoca):
        for tipo in losses:
            for loss in losses[tipo]:
                self.tb.add_scalar('training ' + loss, losses[tipo][loss], epoca)

    #comparação entre as imagens originais e reconstruidas
    def add_img_comp(self, real, fake, epoca, name="amostra"):
        join_real = make_grid(desnormalize(real.to('cpu')), real.size(0))

        if type(fake) == list:
            for i in range(len(fake)):
                fake[i] = make_grid(desnormalize(fake[i].to('cpu')), fake[i].size(0))
            join_fake = fake
        else:
            join_fake = [make_grid(desnormalize(fake.to('cpu')), fake.size(0))]

        final = make_grid([join_real, *join_fake], 1)

        self.tb.add_image(
            name + '_' + str(epoca),
            final
        )

        if not os.path.isdir(self.path_snaps):
            os.makedirs(self.path_snaps)

        save_img(final, os.path.join(self.path_snaps, str(epoca) + '.png'))