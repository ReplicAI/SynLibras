import torch
import torch.nn.functional as F

def divKLPrior(mu, logvar, pmu, plogvar):        
    var = torch.exp(logvar)
    pvar = torch.exp(plogvar)
    
    return torch.mean(0.5 * torch.sum( (plogvar - logvar) - 1 + (var/pvar) + ((pmu-mu)**2 / pvar), dim=1))

def divKL(mu, logvar):
    kl = -0.5 * (1 + logvar - logvar.exp() - mu.pow(2)).sum(1)

    return torch.mean(kl)

def rec_loss(img, target):
    bs = img.size(0)

    img = img.view(bs, -1)
    target = target.view(bs, -1)

    loss = F.l1_loss(img, target, reduction='none')

    loss = loss.sum(1)

    return loss.mean()

def gan_loss(pred, target):
    bs = pred.size(0)

    if target:
        return F.softplus(-pred).view(bs, -1).mean(dim=1)

    return F.softplus(pred).view(bs, -1).mean(dim=1)