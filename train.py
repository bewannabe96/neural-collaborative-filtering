import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from torch import nn
import torch.utils.data
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import evaluation
from dataset.read import read_multiple_csv_as_df
from model.neumf import NeuMF
from dataset.train_dataset import TrainDataset
from dataset.validation_dataset import ValidationDataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--backend', type=str, default='gloo'),
    parser.add_argument('--hosts', type=str, default=os.environ['SM_HOSTS']),
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST']),

    parser.add_argument('--train-data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-data-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    # model definition
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--mlp-layer', type=int, default=4)
    parser.add_argument('--predictive-factor', type=int, default=64)

    # dataset configuration
    parser.add_argument('--negative-sample-ratio', type=int, default=4)

    # hyper-parameters
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)

    # evaluation property
    parser.add_argument('--eval-k', type=int, default=10)

    return parser.parse_args()


def create_datasets(train_data_dir, validation_data_dir, negative_sample_ratio):
    train_filenames = glob.glob(train_data_dir + '/*.csv')
    validation_filenames = glob.glob(validation_data_dir + '/*.csv')

    train_df = read_multiple_csv_as_df(train_filenames, header=None)
    validation_df = read_multiple_csv_as_df(validation_filenames, header=None)

    # ==============================

    train_max_user, train_max_item = train_df.max()
    validation_max_user, validation_max_item = validation_df.iloc[:, [0, 1]].max()

    user_number = max(train_max_user, validation_max_user)
    item_number = max(train_max_item, validation_max_item)

    # ==============================

    train_dataset = TrainDataset(user_number, item_number,
                                 train_df.iloc[:, 0].values.astype(np.int32),
                                 train_df.iloc[:, 1].values.astype(np.int32),
                                 negative_sample_ratio)

    # ==============================

    sparse_neg_items = np.empty(
        shape=(len(validation_df), len(validation_df.iloc[0, 2].split('|'))),
        dtype=np.int32
    )

    for i, row in enumerate(validation_df.iloc[:, 2]):
        sparse_neg_items[i, :] = np.fromiter((x for x in row.split('|')), dtype=np.int32)

    validation_dataset = ValidationDataset(validation_df.iloc[:, 0].values.astype(np.int32),
                                           validation_df.iloc[:, 1].values.astype(np.int32),
                                           sparse_neg_items)

    return user_number, item_number, train_dataset, validation_dataset


def train(model, train_loader, val_loader, model_dir, tensorboard_log_dir, device=torch.device('cpu'),
          epochs=10, lr=0.01, evaluation_k=10):
    best_norm_dcg = 0

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # before train validation
    model.eval()
    with torch.no_grad():
        hr, mean_ap, norm_dcg = evaluation.evaluate(model, val_loader, evaluation_k, device=device)
        summary_writer.add_scalar("HR@" + str(evaluation_k) + "/validation", hr, 0)
        summary_writer.add_scalar("mAP@" + str(evaluation_k) + "/validation", mean_ap, 0)
        summary_writer.add_scalar("nDCG@" + str(evaluation_k) + "/validation", norm_dcg, 0)

    # start train
    for epoch in range(epochs):
        start_time = time.time()

        batch = 0
        loss_sum = 0
        model.train()
        train_loader.dataset.regenerate_negative_samples()
        for user, item, label in train_loader:
            batch += 1

            user = user.to(device)
            item = item.to(device)
            label = label.reshape(-1, 1).to(device)

            model.zero_grad()
            outputs = model(user, item)

            loss = criterion(outputs, label)
            loss_sum += loss.item()

            loss.backward()
            optimizer.step()

        summary_writer.add_scalar("Loss/train", loss_sum / batch, epoch + 1)

        model.eval()
        with torch.no_grad():
            hr, mean_ap, norm_dcg = evaluation.evaluate(model, val_loader, evaluation_k, device=device)
            summary_writer.add_scalar("HR@" + str(evaluation_k) + "/validation", hr, epoch + 1)
            summary_writer.add_scalar("mAP@" + str(evaluation_k) + "/validation", mean_ap, epoch + 1)
            summary_writer.add_scalar("nDCG@" + str(evaluation_k) + "/validation", norm_dcg, epoch + 1)

        elapsed_time = time.time() - start_time
        summary_writer.add_scalar("train-time", elapsed_time, epoch + 1)

        if norm_dcg > best_norm_dcg:
            best_norm_dcg = norm_dcg

            if 'RANK' not in os.environ or os.environ['RANK'] == '0':
                torch.save(model.state_dict(), model_dir)


if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device(args.device)

    user_number, item_number, train_dataset, validation_dataset = create_datasets(
        args.train_data_dir, args.validation_data_dir, args.negative_sample_ratio
    )

    model = NeuMF(
        args.model_name,
        user_number, item_number,
        predictive_factor_num=args.predictive_factor,
        mlp_layer_num=args.mlp_layer, dropout_prob=0.3
    ).to(device=device)
    print(model)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    hosts = json.loads(args.hosts)
    if len(hosts) > 1:
        world_size = len(hosts)
        rank = hosts.index(args.current_host)

        dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)

        model = torch.nn.parallel.DistributedDataParallel(model)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   sampler=torch.utils.data.distributed.DistributedSampler(
                                                       train_dataset))
        val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                                 sampler=torch.utils.data.distributed.DistributedSampler(validation_dataset))

    train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_dir=os.path.join(args.model_dir, 'model.pth'),
        tensorboard_log_dir=os.path.join(args.output_dir, "tensorboard"),
        epochs=args.epochs, lr=args.lr,
        device=device, evaluation_k=args.eval_k,
    )
