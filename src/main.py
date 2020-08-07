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
from utils.evaluation import ndcg_score, mrr_score
from sklearn.metrics import roc_auc_score


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
@click.option('--out', type=str, default='../out')
def main(data, out):
    # paths
    trn_data = os.path.join(data, 'MINDdemo_train')
    vld_data = os.path.join(data, 'MINDdemo_dev')
    util_data = os.path.join(data, 'utils')
    trn_paths = set_paths(trn_data)
    vld_paths = set_paths(vld_data)
    util_paths = set_utils(util_data)
    out_path = os.path.join(out, 'MINDdemo_dev')
    os.makedirs(out_path, exist_ok=True)

    # read configuration file
    config = prepare_config(util_paths['config'],
                            wordEmb_file=util_paths['embedding'],
                            wordDict_file=util_paths['word_dict'],
                            userDict_file=util_paths['uid2index'])

    # set
    seed = config['seed']
    set_seed(seed)
    epochs = config['epochs']
    metrics = {metric: 0. for metric in config['metrics']}

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
    vld_impr_idx, vld_his, vld_impr, vld_label =\
        vld_set.raw_impr_idxs, vld_set.histories_words, vld_set.imprs_words, vld_set.labels

    # define models, optimizer, loss
    # TODO: w2v --> BERT model
    word2vec_emb = np.load(config['wordEmb_file'])
    model = NRMS(config, word2vec_emb).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']),
                           weight_decay=float(config['weight_decay']))
    criterion = nn.CrossEntropyLoss()

    # train and evaluate
    for epoch in range(1, epochs+1):
        start_time = time.time()
        batch_loss = 0.
        '''
        training
        '''
        for i, (trn_his, trn_pos, trn_neg) in enumerate(trn_loader):
            # ready for training
            model.train()
            optimizer.zero_grad()

            # prepare data
            trn_his, trn_pos, trn_neg = \
                trn_his.to(DEVICE), trn_pos.to(DEVICE), trn_neg.to(DEVICE)
            trn_cand = torch.cat((trn_pos, trn_neg), dim=1)
            trn_gt = torch.zeros(size=(trn_cand.shape[0],)).long().to(DEVICE)

            # inference
            trn_his_out = model(trn_his, source='history')
            trn_cand_out = model(trn_cand, source='candidate')
            prob = torch.matmul(trn_cand_out, trn_his_out.unsqueeze(2)).squeeze()

            # training
            loss = criterion(prob, trn_gt)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()

        '''
        evaluation
        '''
        with open(os.path.join(out_path, f'prediction-{epoch}.txt'), 'w') as f:
            for j in range(len(vld_impr)):
                impr_idx_j = vld_impr_idx[j]
                vld_his_j = torch.tensor(vld_his[j]).long().to(DEVICE).unsqueeze(0)
                vld_his_out_j = model(vld_his_j, source='history')
                vld_cand_j = torch.tensor(vld_impr[j]).long().to(DEVICE).unsqueeze(0)
                vld_cand_out_j = model(vld_cand_j, source='candidate')

                scores_j = torch.matmul(vld_cand_out_j, vld_his_out_j.unsqueeze(2)).squeeze()
                sorted_j = torch.sort(scores_j, descending=True)[1] + 1
                scores_j = scores_j.detach().cpu().numpy()
                sorted_j = list(sorted_j.detach().cpu().numpy())
                sorted_idx = ','.join([str(s) for s in sorted_j])
                f.write(f'{impr_idx_j} [{sorted_idx}]\n')

                vld_gt_j = np.array(vld_label[j])

                for metric, _ in metrics.items():
                    if metric == 'auc':
                        score = roc_auc_score(vld_gt_j, scores_j)
                        metrics[metric] += score
                    elif metric == 'mrr':
                        score = mrr_score(vld_gt_j, scores_j)
                        metrics[metric] += score
                    elif metric.startswith('ndcg'):  # format like: ndcg@5;10
                        k = int(metric.split('@')[1])
                        score = ndcg_score(vld_gt_j, scores_j, k=k)
                        metrics[metric] += score

        for metric, _ in metrics.items():
            metrics[metric] /= len(vld_impr)

        epoch_loss = batch_loss/(i+1)
        end_time = time.time()

        result = f'Epoch {epoch:3d} [{end_time-start_time:5.2f}], TrnLoss:{epoch_loss:.4f}'
        result += ', '
        for enum, (metric, _) in enumerate(metrics.items(), start=1):
            result += f'{metric}:{metrics[metric]:.4f}'
            if enum < len(metrics):
                result += ', '
        print(result)


if __name__ == '__main__':
    main()