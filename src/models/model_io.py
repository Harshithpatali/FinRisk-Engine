import torch
import os


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path: str):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
