import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data',
        nargs='+',
        help='Training file location',
        default='gs://rnssnce/new/*.npy')
    parser.add_argument(
        '--job_dir',
        type=str,
        help='Dir to write checkpoints and export model',
        default='gs://rnssnce')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training steps')
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=15,
        help='Maximum number of epochs on which to train')
    parser.add_argument(
        '--checkpoint_epochs',
        type=int,
        default=2,
        help='Checkpoint per n training epochs')
    parser.add_argument(
        '--data_samples',
        type=int,
        default=10000,
        help='Num of samples per class'
    )

    args, _ = parser.parse_known_args()
    return args
