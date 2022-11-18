from .phoenix import Phoenix
import torch, warnings

def create_dataset(opt):
    dataset = ConfigurableDataLoader(opt)
    return dataset

class ConfigurableDataLoader:
    def __init__(self, opt):
        self.opt = opt

        dataset = Phoenix(self.opt)

        shuffle = self.opt.shuffle

        print("[/] Dataset [%s] of size %d was created. shuffled=%s" % (type(dataset).__name__, len(dataset), shuffle))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size = self.opt.batch_size,
                shuffle = shuffle,
                num_workers = int(self.opt.num_workers),
                drop_last=True
            )

            self.dataloader_iterator = iter(self.dataloader)

        self.length = len(dataset)

    def __iter__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.dataloader_iterator = iter(self.dataloader)

        return self

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        return next(self.dataloader_iterator)