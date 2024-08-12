import torch


def get_hits(match_indices):
    return torch.where(match_indices >= 0, 1.0, 0.0).squeeze()


def get_avg_precisions(match_indices):
    avg_precision = torch.reciprocal(match_indices + 1)
    return torch.where(avg_precision == torch.inf, 0, avg_precision).squeeze()


def get_norm_dcg(match_indices):
    dcg = torch.reciprocal(torch.log2(match_indices + 2))
    return torch.where(dcg == torch.inf, 0, dcg).squeeze()


def evaluate(_model, dataset_loader, k, device=torch.device('cpu')):

    hits = torch.tensor([]).to(device)
    avg_precisions = torch.tensor([]).to(device)
    norm_dcgs = torch.tensor([]).to(device)

    for user, pos_item, neg_items in dataset_loader:
        neg_size = neg_items.size(1)
        item_size = 1 + neg_size

        user = user.reshape(-1, 1).repeat_interleave(item_size, dim=1).to(device)
        item = torch.cat((pos_item.reshape(-1, 1), neg_items), dim=1).to(device)

        user = user.reshape(-1)
        item = item.reshape(-1)

        output = _model(user, item).reshape(-1, item_size)
        _, top_indices = torch.topk(output, k)

        item = item.reshape(-1, item_size)
        recommendation = item[torch.arange(item.size(0)).unsqueeze(1), top_indices]
        pos_item = pos_item.reshape(-1, 1).to(device)

        mask = recommendation == pos_item
        indices = torch.arange(k).to(device)
        match_indices = torch.where(mask, indices, -1).max(dim=1, keepdim=True).values

        hits = torch.cat((hits, get_hits(match_indices)))
        avg_precisions = torch.cat((avg_precisions, get_avg_precisions(match_indices)))
        norm_dcgs = torch.cat((norm_dcgs, get_norm_dcg(match_indices)))

    return torch.mean(hits).item(), torch.mean(avg_precisions).item(), torch.mean(norm_dcgs).item()
