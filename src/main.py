import os
import pickle
import time

import click
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from models.nrms import NRMS
from utils.config import prepare_config
from utils.dataloader import DataSetTrn, DataSetTest

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_paths(data):
    paths = {'behaviors': os.path.join(data, 'behaviors.tsv'),
             'news': os.path.join(data, 'news.tsv'),
             'entity': os.path.join(data, 'entity_embedding.vec'),
             'relation': os.path.join(data, 'relation_embedding.vec')}
    return paths


def set_utils(data):
    paths = {'embedding': os.path.join(data, 'embedding.npy'),
             'uid2index': os.path.join(data, 'uid2index.pkl'),
             'word_dict': os.path.join(data, 'word_dict.pkl'),
             'config': os.path.join(data, 'nrms.yaml')}
    return paths


def load_dict(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@click.command()
@click.option('--data', type=str, default='/data/mind')
def main(data):
    trn_data = os.path.join(data, 'MINDdemo_train')
    vld_data = os.path.join(data, 'MINDdemo_dev')
    util_data = os.path.join(data, 'utils')
    trn_paths = set_paths(trn_data)
    vld_paths = set_paths(vld_data)
    util_paths = set_utils(util_data)

    # read configuration file
    config = prepare_config(util_paths['config'],
                            wordEmb_file=util_paths['embedding'],
                            wordDict_file=util_paths['word_dict'],
                            userDict_file=util_paths['uid2index'])

    # set
    seed = config['seed']
    set_seed(seed)
    epochs = config['epochs']

    # load dictionaries
    word2idx = load_dict(config['wordDict_file'])
    uid2idx = load_dict(config['userDict_file'])

    # load datasets and define dataloaders
    trn_set = DataSetTrn(trn_paths['news'], trn_paths['behaviors'],
                         word2idx=word2idx, uid2idx=uid2idx, config=config)
    vld_set = DataSetTest(vld_paths['news'], vld_paths['behaviors'],
                          word2idx=word2idx, uid2idx=uid2idx, config=config,
                          label_known=True)
    trn_loader = DataLoader(trn_set, batch_size=config['batch_size'],
                            shuffle=True, num_workers=8)
    vld_his, vld_impr, vld_label = vld_set.histories_words, vld_set.imprs_words, vld_set.labels

    # TODO: w2v --> BERT model
    word2vec_emb = np.load(config['wordEmb_file'])
    model = NRMS(config, word2vec_emb).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']),
                           weight_decay=float(config['weight_decay']))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        start_time = time.time()
        batch_loss = 0.
        for i, (his, pos, neg) in enumerate(trn_loader):
            # ready
            model.train()
            optimizer.zero_grad()

            # prepare data
            his, pos, neg = his.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            cand = torch.cat((pos, neg), dim=1)
            gt = torch.zeros(size=(cand.shape[0],)).long().to(DEVICE)

            # inference
            his_out = model(his, source='history')
            cand_out = model(cand, source='candidate')
            prob = torch.matmul(cand_out, his_out.unsqueeze(2)).squeeze()

            # training
            loss = criterion(prob, gt)
            loss.backward()
            optimizer.step()

            # evaluate
            batch_loss += loss.item()

        epoch_loss = batch_loss/(i+1)
        end_time = time.time()
        print(f'Epoch {epoch:3d} [{end_time-start_time:5.2f}], TrnLoss:{epoch_loss:.4f}')


if __name__ == '__main__':
    main()