import torch
import argparse
import os
import time
import numpy as np
import random

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverffts import FedFFTS
from models.ffts_model import FFTSModel
from utils.mem_utils import MemReporter


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model
    
    for i in range(args.prev, args.times):

        fix_seed = 42
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()


        # Create model based on algorithm
        if args.algorithm == "FFTS":
            args.model = FFTSModel(
                feat_dim=1,
                max_len=args.max_seq_len,
                d_model=args.d_model,
                n_heads=args.n_heads,
                num_layers=args.e_layers,
                patch_len=getattr(args, 'patch_len', 16),
                stride=getattr(args, 'stride', 8),
                dropout=getattr(args, 'dropout', 0.1),
                activation=getattr(args, 'activation', 'gelu'),
                num_experts=getattr(args, 'num_experts', 4),
                top_k=args.topk_value
            ).to(args.device)

        # Generate FL framework
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        elif args.algorithm == "FFTS":
            server = FedFFTS(args, i)
        else:
            raise NotImplementedError

        server.train()
        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    # Calculate average metrics
    # average_data(configs=args, dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser(description='Federated Foundation Model for Time Series')


    # basic
    parser.add_argument('--task', type=str, required=True, default='pretrain',
                        help='task name', choices=['long_term_forecast', 'short_term_forecast', 'imputation', 'classification', 'anomaly_detection', 'pretrain', 'pretrain_m4', 'pretrain_m4_full', 'pretrain_long'])
    parser.add_argument('--dataset', type=str, default='weather', help='the dataset')
    parser.add_argument('--task_note', type=str, required=True, default='a temporal note of task')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--eval_gap', type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('--model', type=str, default='Transformer', help='local model name')
    parser.add_argument('--prev', type=int, default=0, help="Previous Running times")
    parser.add_argument('--times', type=int, default=1, help="Running times")
    parser.add_argument('--topk_value', type=int, default=1, help="TopK value in MoE selection")

    # federated learning (including pFedMe / PerAvg / FedProx / FedAMP / FedPHP)
    parser.add_argument('--algorithm', type=str, required=True, default='FedAvg')
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")

    # dataset
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # differential privacy
    parser.add_argument('--privacy', type=bool, default=False, help="differential privacy")
    parser.add_argument('--dp_sigma', type=float, default=0.0)
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--step_size', type=int, default=512, help='step size of data sliding windows operation')
    parser.add_argument('--num_experts', type=int, default=4, help='number of experts')

    # model define (for deep learning models from Time-Series-Library)
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')

    # model define (for pretraining Transfomer Encoder)
    parser.add_argument('--max_seq_len', type=int, default=192, help='Maximum input sequence length. Determines size of transformer layers')
    parser.add_argument('--activation', choices=['relu', 'gelu'], default='gelu', help='activation function')
    parser.add_argument('--pos_encoding', choices=['fixed', 'learnable'], default='fixed', help='embedding methods')
    parser.add_argument('--norm_layer', choices=['BatchNorm', 'LayerNorm'], default='BatchNorm', help='Normalization layer')

    # pretraining conifg
    parser.add_argument('--masking_ratio', type=float, default=0.15,
                        help='Imputation: mask this proportion of each variable')
    parser.add_argument('--mean_mask_length', type=float, default=6,
                        help="Imputation: the desired mean length of masked segments. Used only when `mask_distribution` is 'geometric'.")
    parser.add_argument('--mask_mode', choices={'separate', 'concurrent'}, default='separate',
                        help=("Imputation: whether each variable should be masked separately "
                              "or all variables at a certain positions should be masked concurrently"))
    parser.add_argument('--mask_distribution', choices={'geometric', 'bernoulli'}, default='geometric',
                        help=("Imputation: whether each mask sequence element is sampled independently at random"
                            "or whether sampling follows a markov chain (stateful), resulting in "
                            "geometric distributions of masked squences of a desired mean_mask_length"))

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--num_clients', type=int, default=20, help='as the same as number of datasets')
    parser.add_argument('--global_rounds', type=int, default=100, help='global communication rounds')
    parser.add_argument('--join_ratio', type=float, default=0.5, help='participant ratio in each round')
    parser.add_argument('--local_epochs', type=int, default=5, help='local updating steps')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-2, help='learning rate')
    parser.add_argument('--learning_rate_decay', type=bool, default=False)
    parser.add_argument('--learning_rate_decay_gamma', type=float, default=0.99)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # device
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--device_id', type=str, default='0', help='gpu id')

    # results
    parser.add_argument('--save_folder_name', type=str, default='', help='results save path')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))
    print("=" * 50)

    run(args)

