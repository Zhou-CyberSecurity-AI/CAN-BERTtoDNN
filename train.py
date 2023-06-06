import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from model import SimpleLSTM, SimpleDNN
import torch.utils.data as data

from tensorboardX import SummaryWriter
from tqdm import tqdm
from args import get_train_args
from utils import get_save_dir, get_logger, get_available_devices, load_model, CheckpointSaver, buildDataset, EMA
import utils
from json import dumps


def main(args):
    # args.save_dir = ./save/, args.name = DoS
    args.save_dir = get_save_dir(args.save_dir, args.name, training=True)
    log = get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))
    
    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Get model
    log.info('Building model...')
    # model = SimpleLSTM(input_dim=args.input_dim, hidden_dim=args.hidden_size, output_dim=args.output_dim
    #                    , n_layers=args.lstm_layers, bidirectional=True, dropout=args.drop_prob)
    model = SimpleDNN(input_dim=args.input_dim, hidden_dim=args.hidden_size, output_dim=args.output_dim, dropout=args.drop_prob)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    # ema = EMA(model, args.ema_decay)
    
    # Get saver
    saver = CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    trainDataset, devDataset, testDataset = buildDataset(args.load_dos_dataset, finetune=False)
    train_loader = data.DataLoader(trainDataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    dev_loader = data.DataLoader(devDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(trainDataset)
    loss_function = nn.CrossEntropyLoss()
    loss_function_1 = utils.FocalLoss(class_num=2)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
            tqdm(total=len(train_loader.dataset)) as progress_bar:
            for feature, label in train_loader:
                feature = feature.to(device)
                label = label.to(device)
                batch_size = feature.size(0)
                optimizer.zero_grad()
                
                # Forward
                log_p1 = model(feature.to(torch.float32))
                loss = loss_function_1(log_p1, label.to(torch.long))
               
                loss_val = loss.item()
                
                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                # ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         CE_Loss=loss_val)
                tbx.add_scalar('train/CE_loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    results = evaluate(model, dev_loader, device)
                    saver.save(step, model, results[args.metric_name], device)
                   

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)


def evaluate(model, data_loader, device):
    nll_meter = utils.AverageMeter()
    model.eval()
    label_dict = []
    pred_dict = []
    loss_function = nn.CrossEntropyLoss()
    loss_function_1 = utils.FocalLoss(class_num=2)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for feature, label in data_loader:
            # Setup for forward
            feature = feature.to(device)
            label = label.to(device)
            batch_size = feature.size(0)

            # Forward
            log_p1 = model(feature.to(torch.float32))
            loss = loss_function_1(log_p1, label.to(torch.int64))
            nll_meter.update(loss.item(), batch_size)

            preds = np.argmax(log_p1.detach().cpu().numpy(), axis=1)
            label = label.detach().cpu().numpy()
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(CEL=nll_meter.avg)

            pred_dict.append(preds)
            label_dict.append(label)
    model.train()
    pred_dict = np.concatenate(pred_dict).squeeze()
    label = np.concatenate(label_dict).squeeze()
    results = utils.eval_dicts(pred_dict, label)
    results_list = {'CEL':nll_meter.avg,
                    'ACC': results['ACC'],
                    'PRE': results['PRE'],
                    'REC': results['REC'],
                    'F1': results['F1'],
                    'FPR': results['FPR'],
                    'FNR': results['FNR']}
    return results_list
    
    
    
if __name__ == '__main__':
    main(get_train_args())