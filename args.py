import argparse

def get_setup_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('pre-process and load intrusion detection dataset')

    add_common_args(parser)

    args = parser.parse_args()

    return args


def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model on intrusion detection system')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=500000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0.01,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=12,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('FNR', 'FPR', 'F1', 'Precision', "Accuracy"),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')

    args = parser.parse_args()

    return args


def get_test_args():
    """Get arguments needed in test.py."""
    parser = argparse.ArgumentParser('Test a trained model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')

    # Require load_path for test.py
    args = parser.parse_args()
    if not args.load_path:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args


def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    parser.add_argument('--dos_eval_evaluate',
                        type=str,
                        default='./save/dos_dev.csv')
    parser.add_argument('--dos_test_evaluate',
                        type=str,
                        default='./save/dos_test.csv')
    parser.add_argument('--fuzzy_eval_evaluate',
                        type=str,
                        default='./save/fuzzy_dev.csv')
    parser.add_argument('--fuzzy_test_evaluate',
                        type=str,
                        default='./save/fuzzy_test.csv')
    parser.add_argument('--rpm_eval_evaluate',
                        type=str,
                        default='./save/rpm_dev.csv')
    parser.add_argument('--rpm_test_evaluate',
                        type=str,
                        default='./save/rpm_test.csv')
    parser.add_argument('--gear_eval_evaluate',
                        type=str,
                        default='./save/gear_dev.csv')
    parser.add_argument('--gear_test_evaluate',
                        type=str,
                        default='./save/gear_test.csv')
    parser.add_argument('--normal_eval_evaluate',
                        type=str,
                        default='./save/normal_dev.csv')
    parser.add_argument('--normal_test_evaluate',
                        type=str,
                        default='./save/normal_test.csv')
    
    # load dataset
    parser.add_argument('--load_dos_dataset',
                        type=str,
                        default='./Dataset/DoS/')
    parser.add_argument('--load_fuzzy_dataset',
                        type=str,
                        default='./Dataset/Fuzzy/')
    parser.add_argument('--load_gear_dataset',
                        type=str,
                        default='./Dataset/Gear/')
    parser.add_argument('--load_rpm_dataset',
                        type=str,
                        default='./Dataset/RPM/')
    parser.add_argument('--load_normal_dataset',
                        type=str,
                        default='./Dataset/Normal/')

    


def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=False,
                        default='baseline',
                        help='Name to identify training or test run.')
    parser.add_argument('--input_dim',
                        type=int,
                        default=16,
                        help='dim for input feature.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/distill/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=64,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--output_dim',
                        type=int,
                        default=2,
                        help='Number of features in output layers.')
    parser.add_argument('--lstm_layers',
                        type=int,
                        default=2,
                        help='Number of lstm layers.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')

def get_teacher_args():
    parser = argparse.ArgumentParser('pretrain model for intrusion detection system')

    add_common_args(parser)
    
    parser.add_argument('--pretrain_epoach',
                        type=int,
                        default=3,
                        help='Number of pretrain epoach')
    parser.add_argument('--pretrain_batch_size',
                        type=int,
                        default=128,
                        help='Number of pretrain batch-size')

    parser.add_argument('--pretrian_save_dir',
                        type=str,
                        default='./save/pretrain/',
                        help='Base directory for saving information.')
    
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=False,
                        default='baseline',
                        help='Name to identify training or test run.')
    
    parser.add_argument('--pretrain_seed',
                        type=int,
                        default=1992,
                        help='Random seed for reproducibility.')
    
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=10,
                        help='Maximum number of checkpoints to keep on disk.')
    
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('FNR', 'FPR', 'F1', 'Precision', "Accuracy"),
                        help='Name of dev metric to determine best checkpoint.')
    
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='Number of sub-processes to use per data loader.')
    
    parser.add_argument('--eval_steps',
                        type=int,
                        default=100000,
                        help='Number of steps between successive evaluations.')
    
    parser.add_argument('--bert_output_dim',
                        type=int,
                        default=128,
                        help='Number of steps between successive evaluations.')
    
    parser.add_argument('--output_dim',
                        type=int,
                        default=2,
                        help='Number of features in output layers.')
    
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    args = parser.parse_args()

    return args
