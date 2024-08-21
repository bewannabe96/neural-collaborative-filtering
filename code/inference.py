import json
import logging
import os
import sys

import torch

from model.neumf import NeuMF

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def model_fn(model_dir):
    model = NeuMF.load(
        'neumf',
        torch.load(os.path.join(model_dir, "model.pth"), map_location=DEVICE, weights_only=True),
        dropout_prob=0.3,
    ).to(device=DEVICE)

    model.eval()

    return model


def input_fn(request_body, request_content_type):
    if request_content_type != 'application/json':
        raise ValueError(f'Request content type must be application/json')

    parsed_request_body = json.loads(request_body)

    user_seq_ids = []
    item_seq_ids = []
    for record in parsed_request_body:
        user_seq_ids.extend([record['userId']] * len(record['itemIds']))
        item_seq_ids.extend(record['itemIds'])

    user_tensor = torch.tensor(user_seq_ids).to(DEVICE)
    item_tensor = torch.tensor(item_seq_ids).to(DEVICE)

    return user_seq_ids, item_seq_ids, user_tensor, item_tensor


def predict_fn(input_data, model):
    model = model
    user_seq_ids, item_seq_ids, user_tensor, item_tensor = input_data

    ignorable = torch.logical_or(
        user_tensor > model.user_number,
        item_tensor > model.item_number,
    )

    user_tensor[ignorable] = 1
    item_tensor[ignorable] = 1

    output = model(user_tensor, item_tensor).squeeze(1)
    output = torch.sigmoid(output) * 100
    output[ignorable] = 0

    return user_seq_ids, item_seq_ids, output.tolist()


def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError(f"Accept type {accept} is not supported")

    user_seq_ids, item_seq_ids, p_ctrs = prediction

    output = []
    last = None
    for user_seq_id, item_seq_id, p_ctr in zip(user_seq_ids, item_seq_ids, p_ctrs):
        if last is None or last != user_seq_id:
            output.append({'userId': user_seq_id, 'items': []})

        last = user_seq_id
        output[-1]['items'].append({'id': item_seq_id, 'pCtr': p_ctr})

    return json.dumps(output), accept
