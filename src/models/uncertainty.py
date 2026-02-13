import torch


def mc_dropout_predict(model, x, n_samples=50):
    """
    Monte Carlo Dropout for uncertainty estimation.
    """

    model.train()  # enable dropout
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred.unsqueeze(0))

    predictions = torch.cat(predictions, dim=0)

    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    return mean, std
