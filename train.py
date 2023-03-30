""" train.py: Trains XLMRobertaExtModel model using configuration set in config.py """

import os
import argparse
import torch
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import XLMRobertaExtDataset
from model import XLMRobertaExtModel
from utils import ExtEvaluator
from config import get_cfg
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus')
    args = parser.parse_args()
    args.world_size = args.gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29600'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=gpu)
    
    cfg = get_cfg()
    
    # Load dataset
    train_dataset = XLMRobertaExtDataset(fpath=cfg['trn_data'],
                                         max_src_sentences=cfg['max_src_sentences'],
                                         pos_embed=cfg['pos_embed'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=gpu)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=cfg['batch_size'],
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    # Load model
    model = XLMRobertaExtModel(pos_embed=cfg['pos_embed'])
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    # Prepare optimizer and utilities
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss(reduction='none')
    num_training_steps = int(round(len(train_loader) * cfg['epochs'] / cfg['accumulations']))

    # Wrap model
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    
    # Init logging
    if gpu == 0:
        evaluator = ExtEvaluator(cfg['eval_data'], cfg['eval_size'], cfg['pos_embed'])
        progress_bar = tqdm(range(num_training_steps))
        wandb.init(project=cfg['wandb_project'],  name=cfg['wandb_run'])
        wandb.watch(model, log_freq=cfg['report_freq'])
        wandb.log({'eval_loss' : evaluator(model.module)})
        
    # Start training
    min_eval_loss = 1e5
    trn_step = 0
    model.train()
    for epoch in range(cfg['epochs']):
        optimizer.zero_grad()
        avg_loss = 0
        step_cnt = 0
        for (i, batch) in enumerate(train_loader):
            batch = {k: v.squeeze(1).cuda(non_blocking=True) for k, v in batch.items()}
            # Compute sentence scores
            scores = model(batch['input_ids'],
                           batch['attention_mask'],
                           batch['cls_ids'],
                           batch['cls_mask'])
            # Compute loss
            loss = loss_fn(scores, batch['labels'])
            loss = (loss * batch['cls_mask']).sum() / loss.numel()
            loss = loss / cfg['accumulations']
            loss.backward()
            avg_loss += loss.item()

            # Optimization step
            if (i+1) % cfg['accumulations'] == 0 or (i + 1 == len(train_loader)):
                trn_step += 1
                # Learning rate update
                lr = cfg['max_lr'] * min(trn_step ** (-0.5), trn_step * (cfg['warmup'] ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()
                step_cnt = 0
                if gpu == 0:
                    progress_bar.update(1)

            # Report step
            if (trn_step+1) % cfg['report_freq'] == 0 and step_cnt == 0 and gpu == 0:
                wandb.log({'loss': avg_loss / cfg['report_freq']})
                wandb.log({'epoch' : trn_step / (num_training_steps / cfg['epochs'])})
                wandb.log({'lr' : lr})
                avg_loss = 0

            # Evaluation step
            if (trn_step+1) % cfg['eval_freq'] == 0 and step_cnt == 0 and gpu == 0:
                eval_loss = evaluator(model.module)
                wandb.log({'eval_loss' : eval_loss})
                # Save model
                if eval_loss < min_eval_loss:
                    model.module.save(cfg['save_path'])
                    wandb.log({'best_model_step' : trn_step})
                    min_eval_loss = eval_loss
            step_cnt += 1


if __name__ == "__main__":
    main()     

