import argparse
import json
import random
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F

from utils import ensure_path, set_gpu, l2_loss, config_logger
from models.gcn import GCN
import logging
import time

def save_checkpoint(name):
    torch.save(gcn.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj, osp.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=3000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save-epoch', type=int, default=300)
    parser.add_argument('--save-path', default='gcn-basic')
    parser.add_argument('--hidden_layers', default='d2048,d')
    parser.add_argument('--k', type=eval, default=(1,1))

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--no-pred', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    set_gpu(args.gpu)

    save_path = args.save_path
    save_path += '-k={}'.format(args.k)
    save_path = osp.join('save', args.trainval, save_path)
    ensure_path(save_path)
    config_logger(save_path+'/train.log', verbose=args.verbose)
    logging.info(args)

    graph = json.load(open('materials/imagenet-induced-graph.json', 'r'))
    wnids = graph['wnids']
    n = len(wnids)
    edges = graph['edges']
    
    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    word_vectors = torch.tensor(graph['vectors']).cuda()
    word_vectors = F.normalize(word_vectors)

    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    assert train_wnids == wnids[:len(train_wnids)]
    fc_vectors = torch.tensor(fc_vectors).cuda()
    fc_vectors = F.normalize(fc_vectors)

    hidden_layers = args.hidden_layers
    gcn = GCN(n, edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers, ks=args.k).cuda()
    logging.info(gcn)
    logging.info('{} nodes, {} edges'.format(n, len(edges)))
    logging.info('word vectors: {}'.format(word_vectors.shape))
    logging.info('fc vectors: {}'.format(fc_vectors.shape))
    logging.info('hidden layers: {}'.format(hidden_layers))

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.trainval == 'awa2':
        awa2_split = json.load(open('materials/awa2-split.json', 'r'))
        train_wnids = awa2_split['train']
        test_wnids = awa2_split['test']

        wnid_dic = dict(zip(wnids, range(len(wnids))))
        tlist = [wnid_dic[wnid] for wnid in train_wnids if wnid_dic[wnid] < 1000]
        v_val = 0
    else:
        v_train, v_val = map(float, args.trainval.split(','))
        n_trainval = len(fc_vectors)
        n_train = round(n_trainval * (v_train / (v_train + v_val)))
        logging.info('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
        tlist = list(range(len(fc_vectors)))
        if args.seed != -1:
            random.shuffle(tlist)
        tlist, vlist = tlist[:n_train], tlist[n_train:]

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0
    train_time = 0

    for epoch in range(1, args.max_epoch + 1):
        gcn.train()
        step_time = time.time()
        output_vectors = gcn(word_vectors)
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_time = time.time() - step_time
        train_time += step_time

        gcn.eval()
        output_vectors = gcn(word_vectors)
        train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist).item()
        if v_val > 0:
            val_loss = mask_l2_loss(output_vectors, fc_vectors, vlist).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss
        logging.debug('epoch {}, train_loss={:.4f}, val_loss={:.4f}, step_time={:.4f}'
              .format(epoch, train_loss, val_loss, step_time))

        trlog['train_loss'].append(train_loss)
        trlog['val_loss'].append(val_loss)
        trlog['min_loss'] = min_loss
        torch.save(trlog, osp.join(save_path, 'trlog'))

        if (epoch % args.save_epoch == 0):
            logging.info('epoch {}, train_loss={:.4f}, val_loss={:.4f}, train_time={:.4f}'
                  .format(epoch, train_loss, val_loss, train_time))
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors
                }

        if epoch % args.save_epoch == 0:
            save_checkpoint('epoch-{}'.format(epoch))
        
        pred_obj = None

