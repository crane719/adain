import torch

def adain(crep, srep):
    cmu = mu(crep)
    smu = mu(srep)
    csigma = sigma(crep)
    ssigma = sigma(srep)
    return ssigma*(crep-cmu)/(csigma+1e-10)+smu

def mu(vec, keep_dim=True):
    if keep_dim:
        return torch.mean(vec.view(vec.shape[0], vec.shape[1], -1), dim=2).unsqueeze(2).unsqueeze(3)
    else:
        return torch.mean(vec.view(vec.shape[0], vec.shape[1], -1), dim=2)

def sigma(vec, keep_dim=True):
    if keep_dim:
        return torch.std(vec.view(vec.shape[0], vec.shape[1], -1), dim=2).unsqueeze(2).unsqueeze(3)
    else:
        return torch.std(vec.view(vec.shape[0], vec.shape[1], -1), dim=2)
