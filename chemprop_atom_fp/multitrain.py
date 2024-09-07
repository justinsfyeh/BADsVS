import torch.multiprocessing as mp
from chemprop.args import TrainArgs
from chemprop.train import cross_validate, run_training
import itertools

def Training(fold, rpt, hyp):
    args = TrainArgs().parse_args()
    depth, dropout, hidden_size, ffn_num_layers = hyp
    args.save_dir = f'{args.save_dir}{str(fold)}_{str(rpt)}'
    args.depth = depth
    args.dropout = dropout
    args.hidden_size = hidden_size
    args.ffn_num_layers = ffn_num_layers

    args.process_args()
    cross_validate(args, train_func=run_training)

if __name__ == '__main__':

    depth = [2,3,4]
    dropout = [0, 0.2, 0.4]
    hidden_size = [500] #[100, 300, 500]
    ffn_num_layers = [4] # [2,3,4]
    hyperparameters = list(itertools.product(depth, dropout, hidden_size, ffn_num_layers))

    num_processes = 9
    # folds = (0, 1, 2)

    processes = []
    mp.set_start_method("spawn")
    # for fold in folds:
    fold = 1
    for rank in range(num_processes):
        use = (fold, rank, hyperparameters[rank])
        p = mp.Process(target=Training, args=use)
        p.start()
        processes.append(p)
            
    for p in processes:
        p.join()