import argparse


def parse():

    ########### cancer #############################

    #
    # # ##########
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    #     description='Implementation of MPO on gym environments')
    #
    # parser.add_argument('--env', type=str, default="CancerEnvironment",
    #                     help='Options: CancerEnvironment, HIVEnvironment, SepsisEnvironment')
    #
    # parser.add_argument('--deg', type=int, default=3,
    #                     help='degree of polynomial')
    #
    # parser.add_argument('--sign', type=float, default=1.0,
    #                     help='sign of attacker\'s objective')
    #
    # parser.add_argument('--feature_type', type=str, default='poly',
    #                     help='type of features - poly/rbf')
    #
    # parser.add_argument('--initial_lr', type=float, default=0.1,
    #                     help='initial learning rate for line search')
    #
    # parser.add_argument('--projection_type', type=str, default='l1',
    #                     help='projection type')
    #
    # parser.add_argument('--epsilon', type=float, default=0.25,
    #                     help='epsilon value')
    #
    # parser.add_argument('--num_centroids', type=int, default=60,
    #                     help='no of centroids in rbf')
    #
    # parser.add_argument('--max_epochs', type=int, default=2000,
    #                     help='max epochs')
    #
    #
    # parser.add_argument('--reg', type=float, default=1e-2,
    #                     help='regularization')
    #
    # parser.add_argument('--eps', type=float, default=0.1,
    #                     help='eps value')
    #
    # parser.add_argument('--random', type=bool, default=False,
    #                     help='random')
    #
    # parser.add_argument('--interactive', type=bool, default=False,
    #                     help='interactive')
    #
    # parser.add_argument('--iters', type=int, default=100,
    #                     help='iters')
    #
    # parser.add_argument('--num_samples', type=int, default=20000,
    #                     help='iters')
    #
    #
    # parser.add_argument('--attacker_type', type=str, default="influence",
    #                     help='attacker type OPTIONS: influence, random')
    #
    # parser.add_argument('--is_cuda', type=bool, default=False,
    #                     help='attacker type')
    #
    # parser.add_argument('--is_type', type=bool, default="cpdis",
    #                     help='IS type')
    #
    # parser.add_argument('--method', type=str, default="FQE",
    #                     help='method name, OPTIONS: FQE, IS, DR')
    #
    # parser.add_argument('--tensor', type=bool, default=True,
    #                     help='True if attacker has to modify original features instead of transformed features')


    ##########
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Implementation of MPO on gym environments')

    parser.add_argument('--env', type=str, default="CustomEnvironment",
                        help='Options: CancerEnvironment, HIVEnvironment, SepsisEnvironment')

    parser.add_argument('--deg', type=int, default=3,
                        help='degree of polynomial')

    parser.add_argument('--sign', type=float, default=1.0,
                        help='sign of attacker\'s objective')

    parser.add_argument('--feature_type', type=str, default='poly',
                        help='type of features - poly/rbf')

    parser.add_argument('--initial_lr', type=float, default=1.0,
                        help='initial learning rate for line search')

    parser.add_argument('--projection_type', type=str, default='l1',
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

    parser.add_argument('--random', type=bool, default=False,
                        help='random')

    parser.add_argument('--interactive', type=bool, default=False,
                        help='interactive')

    parser.add_argument('--iters', type=int, default=100,
                        help='iters')

    parser.add_argument('--num_samples', type=int, default=300000,
                        help='iters')

    parser.add_argument('--attacker_type', type=str, default="influence",
                        help='attacker type OPTIONS: influence, random')

    parser.add_argument('--is_cuda', type=bool, default=False,
                        help='attacker type')

    parser.add_argument('--is_type', type=str, default="pdis",
                        help='IS type')

    parser.add_argument('--method', type=str, default="PDIS",
                        help='method name, OPTIONS: FQE, IS, DR')

    parser.add_argument('--tensor', type=bool, default=True,
                        help='True if attacker has to modify original features instead of transformed features')


    return parser.parse_args()
