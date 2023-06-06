import logging
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import queue
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import Dataset
import sklearn.metrics as sm
from torch.autograd import Variable
import torch.nn as nn
from transformers import BertTokenizer
import pickle

def loss(self, output, bert_prob, real_label):
    criterion = torch.nn.BCEWithLogitsLoss()
    return criterion(output, real_label.float())

def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')
    
def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids

def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)
    # print(ckpt_dict['model_state'])
    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'], strict=False)

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model

class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

def buildDataset(path, finetune=True):
    #数据集替换
    f_read = open('./Dataset/DoS/DataforBertTrain.npy', 'rb')
    data = pickle.load(f_read)
    label = np.load(path+"Label.npy")
    print(label.shape)
    input_ids = data['input_ids']
    attention_mask = data["attention_mask"]
    
    if finetune:
        X_train_ids, X_test_ids, X_train_mask, X_test_mask, Y_train, Y_test = train_test_split(input_ids,attention_mask, label, test_size=0.2, shuffle=True, random_state=2023, stratify=label)
        X_test_ids, X_eval_ids, X_test_mask, X_eval_mask, Y_test, Y_eval = train_test_split(X_test_ids, X_test_mask, Y_test, test_size=0.1, shuffle=True, random_state=2022, stratify=Y_test)
        trainDataset = BertClassDataset(X_train_ids, X_train_mask, Y_train)
        devDataset = BertClassDataset(X_eval_ids, X_eval_mask, Y_eval)
        testDataset = BertClassDataset(X_test_ids, X_test_mask, Y_test)
    else:
        X_train_ids, X_test_ids, Y_train, Y_test = train_test_split(input_ids, label, test_size=0.2, shuffle=True, random_state=2023, stratify=label)
        X_test_ids, X_eval_ids, Y_test, Y_eval = train_test_split(X_test_ids, Y_test, test_size=0.1, shuffle=True, random_state=2022, stratify=Y_test)
        
        trainDataset = SimpleClassDataset(X_train_ids, Y_train)
        devDataset = SimpleClassDataset(X_eval_ids, Y_eval)
        testDataset = SimpleClassDataset(X_test_ids, Y_test)
  
    return trainDataset, devDataset, testDataset

class BertClassDataset(Dataset):
    def __init__(self, ids, mask, label):
        super(BertClassDataset, self).__init__()
        self.input_ids = ids
        self.attention_mask = mask
        self.labels = label

    def __getitem__(self, index):
        return torch.LongTensor(self.input_ids[index]), torch.LongTensor(self.attention_mask[index]), self.labels[index]

    def __len__(self):
        return len(self.labels)
    
class SimpleClassDataset(Dataset):
    def __init__(
            self,
            data,
            labels,
    ):
        super(SimpleClassDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

#更换数据集
def getDataset(path, finetune=True):
    f_read = open('./Dataset/DoS/DataforBertTrain.npy', 'rb')
    dataBert = pickle.load(f_read)
    label = np.load(path+"Label.npy")

   
    input_ids = dataBert['input_ids']  
    attention_mask = dataBert['attention_mask']

    
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, Y_train, Y_test = train_test_split(input_ids, attention_mask, label, test_size=0.2, shuffle=True, random_state=2023, stratify=label)
    X_test_ids, X_eval_ids, X_test_mask, X_eval_mask, Y_test, Y_eval = train_test_split(X_test_ids, X_test_mask, Y_test, test_size=0.1, shuffle=True, random_state=2022, stratify=Y_test)
    
    trainDataset = distillClassDataset(X_train_ids, X_train_mask, Y_train)
    devDataset = distillClassDataset(X_eval_ids, X_eval_mask, Y_eval)
    testDataset = distillClassDataset(X_test_ids, X_test_mask, Y_test)
    return trainDataset, devDataset, testDataset

class distillClassDataset(Dataset):
    def __init__(self, ids, mask, label):
        super(distillClassDataset, self).__init__()
        self.input_ids = ids
        self.attention_mask = mask
        self.labels = label

    def __getitem__(self, index):
        return torch.LongTensor(self.input_ids[index]), torch.LongTensor(self.attention_mask[index]), self.labels[index]

    def __len__(self):
        return len(self.labels)
    
def DistillationLoss(student_logits, teacher_logits, label):
    student_loss = nn.CrossEntropyLoss()
    distillation_loss = nn.KLDivLoss()
    temperature = 1
    alpha = 0.4
    
    student_target_loss = student_loss(student_logits, label)
    distillation_loss = distillation_loss(F.log_softmax(student_logits / temperature, dim=1), F.softmax(teacher_logits / temperature, dim=1))
    loss = (1 - alpha) * student_target_loss + alpha * distillation_loss
    return loss
class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

def eval_dicts(pred_dict, label_dict):
    confusionMatrix = sm.confusion_matrix(pred_dict, label_dict)
    TP, TN, FP, FN = confusionMatrix[1][1], confusionMatrix[0][0], confusionMatrix[1][0], confusionMatrix[0][1]
    ACC = (TP+TN)/(TP+TN+FP+FN)
    PRE = TP/(TP+FP)
    REC = TP/(TP+FN)
    F1 = (2*PRE*REC)/(PRE+REC)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    # ACC = sm.accuracy_score(pred_dict, label_dict)
    # PRE = sm.precision_score(pred_dict, label_dict)
    # REC = sm.recall_score(pred_dict, label_dict)
    # F1 = sm.f1_score(pred_dict, label_dict)
    # FPR = 1-sm.recall_score(pred_dict, label_dict, pos_label=0)
    # FNR = 1-sm.precision_score(pred_dict, label_dict, pos_label=0)
    rerults = {"ACC":ACC, "PRE":PRE,
               "REC":REC, "F1":F1,
               "FPR":FPR, "FNR":FNR}
    return rerults


 
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=3, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
 
 
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
 
        probs = (P*class_mask).sum(1).view(-1,1)
 
        log_p = probs.log()
        # print("log_p:{}".format(log_p))
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
 
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        # print('-----bacth_loss------')
        # print("batch_loss:{}".format(batch_loss))
 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# 更换数据集

if __name__ == "__main__":
    getDataset("./Dataset/DoS/", finetune=True)
    