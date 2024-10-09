import torch
import torch.nn.functional as F


def eval(loader, model, args):

    model.eval()

    correct = 0.
    loss = 0.

    for data in loader:
        data = data.to(args.device)
        with torch.no_grad():
            out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()

    return correct / len(loader.dataset), loss / len(loader.dataset)
