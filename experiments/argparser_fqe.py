import argparse


def parse():

    ########### cancer #############################

    ##########
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Implementation of DOPE')

    parser.add_argument('--env', type=str, default="MountainCarEnvironment",
                        help='Options: CancerEnvironment, HIVEnvironment, MountainCarEnvironment')

    parser.add_argument('--deg', type=int, default=3,
                        help='degree of polynomial')

    parser.add_argument('--seed', type=int, default=1,
                        help='degree of polynomial')

    parser.add_argument('--sign', type=float, default=1.0,
                        help='sign of attacker\'s objective')

    parser.add_argument('--feature_type', type=str, default='rbf',
                        help='type of features - poly/rbf')

    parser.add_argument('--initial_lr', type=float, default=1.0,
                        help='initial learning rate for line search')

    parser.add_argument('--type', type=str, default='linf',
                        help='projection type')

    parser.add_argument('--epsilon', type=float, default=-1.0,
                        help='epsilon value')

    parser.add_argument('--num_centroids', type=int, default=60,
                        help='no of centroids in rbf')

    parser.add_argument('--max_epochs', type=int, default=2000,
                        help='max epochs')

    parser.add_argument('--reg', type=float, default=1e-2,
                        help='regularization')

    parser.add_argument('--eps', type=float, default=0.1,
                        help='eps value')

    parser.add_argument('--random', type=int, default=0,
                        help='random')

    parser.add_argument('--interactive', type=bool, default=False,
                        help='interactive')

    parser.add_argument('--iters', type=int, default=50,
                        help='iters')

    parser.add_argument('--num_samples', type=int, default=2000000,
                        help='iters')

    parser.add_argument('--attacker_type', type=str, default="influence",
                        help='attacker type OPTIONS: influence, random')

    parser.add_argument('--is_cuda', type=bool, default=False,
                        help='attacker type')

    parser.add_argument('--is_type', type=str, default="is",
                        help='IS type')

    parser.add_argument('--method_type', type=str, default="IS",
                        help='method name, OPTIONS: FQE, IS, WDR')

    parser.add_argument('--tensor', type=bool, default=True,
                        help='True if attacker has to modify original features instead of transformed features')

    parser.add_argument('--experiment_id', type=int, default=1,
                        help='experiment id')

    parser.add_argument('--dataset_id', type=int, default=2,
                        help='dataset id')

    parser.add_argument('--frac', type=float, default=1.0,
                        help='fraction of data points')
    return parser.parse_args()
