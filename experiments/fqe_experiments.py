import sys
sys.path.append("experiments/")
from argparser_fqe import parse

args = parse()
import torch

torch.manual_seed(args.seed)
import random

random.seed(args.seed)
import numpy as np

np.random.seed(args.seed)

print("args.seed",args.seed)
sys.path.append("experiments/")

import os
import sys


sys.path.append("../")

sys.path.append("src/")
sys.path.append("algorithms_ope/")
sys.path.append("attack_algorithms/")

from fqe_method import *
from importance_sampling_method import *
from wdr_method import *
from influence_attack import *
sys.path.append("../domains_for_bilevel/")


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print(sys.path)
sys.path.append("../domains_for_bilevel/")


from utils import *



import sys
sys.path.append("src/")
sys.path.append("attack_algorithms/")


from config import *

from cancer_env import *
from hiv_env import *
# from sepsis_env import *
from cartpole_env import *

from mountaincar_env import *

from argparser_fqe import parse

args = parse()


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
#blockPrint()

sys.path.append("domains/")

if args.env=="CancerEnvironment":
    config = cancer_config
    env = CancerEnv(discount_factor=config.gamma)

if args.env=="CustomEnvironment":
    config = custom_config
    env = CustomEnv(discount_factor=config.gamma)

elif args.env=="HIVEnvironment":
    config = hiv_config

    env = HIVEnv(discount_factor=config.gamma)
    env.max_length = config.max_length
elif args.env=="SepsisEnvironment":
    config = sepsis_config

    env = SepsisEnv(discount_factor=config.gamma)

elif args.env=="MountainCarEnvironment":
    config = mountaincar_config

    env = MountainCarEnv(discount_factor=config.gamma)

elif args.env=="CartpoleEnvironment":
    config = cartpole_config

    env = CartpoleEnv(discount_factor=config.gamma)

use_cuda = torch.cuda.is_available()

if args.is_cuda==True:
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

print(config)

budgets = np.arange(0.0,0.51,0.05)
projections = ["l1","l2","linf"]
epss = np.arange(0.02,0.12,0.02)


def experiment1(env,eps=0.05, trials=1):
    # budgets=[1.0,0.4]
    init_returns_policy1 = []

    init_sum=[]
    current_budgets = []
    attacker=None
    index=0
    dataloader=None
    train_errors = []
    test_errors = []
    eps = config.exp_eps
    before_weights=None
    after_weights=None
    for projection in projections:
        dataloader=None

        for budget in budgets:

            for trial in range(trials):


                attacker = InfluenceAttack(method_type=args.method_type, env=env, config=config, sign=args.sign, initial_lr=args.initial_lr,
                                           type=projection, epsilon=args.epsilon,  max_epochs=args.max_epochs, reg=args.reg,eps=eps, random=0, iters=args.iters, attacker_type='influence',is_type=args.is_type,dataloader=dataloader,frac=budget, dataset_id=args.dataset_id)

                if index==0:
                    ret_policy1 = attacker.model.get_initial_return()
                    init_returns_policy1.append(ret_policy1)
                    ret_policy1 = attacker.dataLoader.eval_rets
                    init_returns_policy1.append(ret_policy1)
                    new_train_error, _, __, pbe = attacker.model.get_train_error()
                    new_test_error, _ , _ = attacker.model.get_test_error()
                    train_errors.append(new_train_error.detach().cpu().numpy())
                    test_errors.append(new_test_error.detach().cpu().numpy())
                ret_policy1 = attacker.model.get_initial_return()

                if args.method_type=="IS":
                    before_weights = attacker.model.is_weights

                attacker.attack()

                ret_policy1 = attacker.model.get_initial_return()

                if args.method_type=="IS":
                    after_weights = attacker.model.is_weights


                init_returns_policy1.append(ret_policy1)

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                np.save("logs/experiment1_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(eps) + "_" + str(args.seed) + "_policy1.npy",
                    np.array(init_returns_policy1))


                new_test_error, _ , _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                np.save(
                    "logs/experiment1_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps)  + "_"+ str(args.seed) + "_pbe.npy",
                    np.array(train_errors))
                np.save(
                    "logs/experiment1_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps)  + "_"+ str(args.seed) + "_train.npy",
                    np.array(test_errors))

                if (projection=="l1" and  args.method_type=="IS"):
                    try:
                        if isinstance(before_weights, np.ndarray) is False:
                            before_weights = before_weights.detach().cpu().numpy()
                        if isinstance(after_weights, np.ndarray) is False:
                            after_weights = after_weights.detach().cpu().numpy()
                        influential_indices = attacker.influence_indices
                        if isinstance(after_weights, np.ndarray) is False:
                            influential_indices = influential_indices.detach().cpu().numpy()
                        np.save("logs/experiment1_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                            eps)  + "_"+ str(budget) + "_" + str(args.seed) + "before_weights.npy",before_weights)

                        np.save("logs/experiment1_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                            eps)  + "_"+ str(budget) + "_" + str(args.seed) + "after_weights.npy",after_weights)
                        np.save("logs/experiment1_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                            eps)  + "_"+ str(budget) + "_" + str(args.seed) + "_influential_indices.npy",influential_indices)
                    except:
                        print(budget)
                if budget==0.5 and projection=="l1" and args.dataset_id==1:
                    attacker.save_model(np.array(train_errors),np.array(test_errors),before_weights,after_weights)
                print("pbe", pbe)
                index+=1
            dataloader = attacker.dataLoader


        print(init_returns_policy1)
        print(current_budgets)


def experiment2(env, trials=1):
    init_returns_policy1 = []


    init_sum = []
    current_budgets = []
    attacker = None
    index = 0
    dataloader = None
    train_errors = []
    test_errors = []
    del_returns = []
    budget = config.exp_budget
    for projection in projections:
        dataloader=None


        for eps in epss:

            for trial in range(trials):


                attacker = InfluenceAttack(method_type=args.method_type, env=env, config=config, sign=args.sign, initial_lr=args.initial_lr,
                                type=projection, epsilon=args.epsilon,  max_epochs=args.max_epochs, reg=args.reg,eps=eps, random=0, iters=args.iters, attacker_type='influence',is_type=args.is_type,dataloader=dataloader,frac=budget, dataset_id=args.dataset_id)


                if index==0:
                    ret_policy1 = attacker.model.get_initial_return()
                    del_returns.append(ret_policy1)
                    init_returns_policy1.append(ret_policy1)
                    ret_policy1 = attacker.dataLoader.eval_rets
                    init_returns_policy1.append(ret_policy1)
                    del_returns.append(ret_policy1)
                    new_train_error, _, __, pbe = attacker.model.get_train_error()
                    new_test_error, _ , _ = attacker.model.get_test_error()
                    train_errors.append(new_train_error.detach().cpu().numpy())
                    test_errors.append(new_test_error.detach().cpu().numpy())


                attacker.attack()
                if args.method_type=="FQE":
                    del_rets =attacker.delete_and_recompute_returns(recompute=False)
                    del_returns.append(del_rets)

                    np.save("logs/experiment2_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(budget) + "_" + str(args.seed) + "_deletedreturns.npy",
                            np.array(del_returns))
                    print("deleted returns",del_returns)



                ret_policy1 = attacker.model.get_initial_return()

                init_returns_policy1.append(ret_policy1)

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                # np.save("logs/experiment2_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(budget) + "_" + str(args.seed) + "_policy1.npy",
                #         np.array(init_returns_policy1))

                new_test_error, _ , _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                np.save(
                        "logs/experiment2_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name + "_" + str(
                            budget)  + "_"+ str(args.seed) +  "_policy1.npy" ,
                        np.array(init_returns_policy1))

                new_test_error, _, _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                np.save(
                    "logs/experiment2_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name + "_" + str(
                        budget)  + "_" + str(args.seed) + "_pbe.npy",
                    np.array(train_errors))
                np.save(
                    "logs/experiment2_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name + "_"  + str(
                        budget)  + "_"+ str(args.seed) + "_train.npy",
                    np.array(test_errors))

                #model.save_model(train_errors,test_errors)

                print("pbe", pbe)
                index += 1
            dataloader = attacker.dataLoader
        print(init_returns_policy1)
        print(current_budgets)



def experiment3(trials=1,eps=0.05):
    discount_factors = [0.2,0.4,0.6,0.8,0.98] #[0.98]
    init_returns_policy1 = []
    before_attack=[]


    init_sum = []
    current_budgets = []
    attacker = None
    index = 0
    dataloader = None
    budget=config.exp_budget
    eps = config.exp_eps
    for projection in projections:

        train_errors = []
        test_errors = []


        for discount in discount_factors:

            if args.env == "CancerEnvironment":
                env = CancerEnv(discount_factor=discount)
                env.max_length = config.max_length

            elif args.env == "CustomEnvironment":
                env = CustomEnv(discount_factor=discount)
                env.max_length = config.max_length

            elif args.env == "HIVEnvironment":
                env = HIVEnv(discount_factor=discount)
                env.max_length = config.max_length

            elif args.env == "MountainCarEnvironment":
                env = MountainCarEnv(discount_factor=discount)
                env.max_length = config.max_length


            elif args.env == "CartpoleEnvironment":
                env = CartpoleEnv(discount_factor=discount)
                env.max_length = config.max_length

            else:
                print("environment not found!")
                exit(1)

            env.discount_factor = discount




            attacker = InfluenceAttack(method_type=args.method_type, env=env, config=config, sign=args.sign, initial_lr=args.initial_lr,
                                       type=projection, epsilon=args.epsilon,  max_epochs=args.max_epochs, reg=args.reg,eps=eps, random=0, iters=args.iters, attacker_type='influence',is_type=args.is_type,dataloader=None,frac=budget, dataset_id=args.dataset_id)


            ret_policy1 = attacker.dataLoader.eval_rets
            before_attack.append(ret_policy1)
            ret_policy1 = attacker.model.get_initial_return()
            before_attack.append(ret_policy1)

            if index==0:
                ret_policy1 = attacker.model.get_initial_return()
                init_returns_policy1.append(ret_policy1)
                ret_policy1 = attacker.dataLoader.eval_rets
                init_returns_policy1.append(ret_policy1)
                new_train_error, _, __, pbe = attacker.model.get_train_error()
                new_test_error, _ , _ = attacker.model.get_test_error()
                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())



            attacker.attack()

            ret_policy1 = attacker.model.get_initial_return()

            init_returns_policy1.append(ret_policy1)

            print("returns policy 1", init_returns_policy1)
            print("budget", budget)
            print("projection", projection)

            np.save("logs/experiment3_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(eps) + "_" + str(args.seed) + "_policy1.npy",
                    np.array(init_returns_policy1))


            new_test_error, _ , _ = attacker.model.get_test_error()

            # get new train error
            new_train_error, _, __, pbe = attacker.model.get_train_error()

            train_errors.append(new_train_error.detach().cpu().numpy())
            test_errors.append(new_test_error.detach().cpu().numpy())

            print("returns policy 1", init_returns_policy1)
            print("budget", budget)
            print("projection", projection)


            np.save(
                "logs/experiment3_before_attack_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                    eps) + "_" + str(budget)   + "_"+ str(args.seed) +"_policy1.npy",
                np.array(before_attack))

            np.save(
                "logs/experiment3_after_attack_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                    eps) + "_" + str(budget)  + "_" +str(args.seed) + "_policy1.npy",
                np.array(init_returns_policy1))

            new_test_error, _, _ = attacker.model.get_test_error()

            # get new train error
            new_train_error, _, __, pbe = attacker.model.get_train_error()

            train_errors.append(new_train_error.detach().cpu().numpy())
            test_errors.append(new_test_error.detach().cpu().numpy())

            np.save(
                "logs/experiment3_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                    eps) + "_" + str(discount) + "_" +str(args.seed) + "_pbe.npy",
                np.array(train_errors))
            np.save(
                "logs/experiment3_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                    eps)+ "_" + str(discount)  + "_" + str(args.seed) +"_train.npy",
                np.array(test_errors))

            print("pbe", pbe)
            index += 1

        print(init_returns_policy1)
        print(current_budgets)



def experiment4(env,eps=0.05, trials=100):

    # budgets = [0.5,0.4]


    init_returns_policy1 = []




    init_sum=[]
    current_budgets = []
    attacker=None
    index=0
    dataloader=None
    train_errors = []
    test_errors = []
    eps = config.exp_eps
    for projection in projections:

        for budget in budgets:
            dataloader=None


            for trial in range(trials):


                attacker = InfluenceAttack(method_type=args.method_type, env=env, config=config, sign=args.sign, initial_lr=args.initial_lr,
                                           type=projection, epsilon=args.epsilon,  max_epochs=args.max_epochs, reg=args.reg,eps=eps, random=0, iters=args.iters, attacker_type="random",is_type=args.is_type,dataloader=dataloader,frac=budget, dataset_id=args.dataset_id)


                if index==0:
                    ret_policy1 = attacker.model.get_initial_return()
                    init_returns_policy1.append(ret_policy1)
                    ret_policy1 = attacker.dataLoader.eval_rets
                    init_returns_policy1.append(ret_policy1)
                    # index+=1

                attacker.attack()

                ret_policy1 = attacker.model.get_initial_return()

                init_returns_policy1.append(ret_policy1)

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                np.save("logs/experiment4_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(eps) + "_" + str(args.seed) + "_policy1.npy",
                        np.array(init_returns_policy1))


                new_test_error, _ , _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)


                np.save(
                    "logs/experiment4_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps)  + "_" + str(args.seed) + "_pbe.npy",
                    np.array(train_errors))
                np.save(
                    "logs/experiment4_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps)  + "_" +str(args.seed) + "_train.npy",
                    np.array(test_errors))

                #model.save_model(train_errors,test_errors)
                print("pbe", pbe)
                index+=1
                attacker.dataLoader.reset()
                dataloader = attacker.dataLoader


        print(init_returns_policy1)
        print(current_budgets)


def experiment5(env,eps=0.05, trials=20):

    init_returns_policy1 = []



    init_sum=[]
    current_budgets = []
    attacker=None
    index=0
    dataloader=None
    train_errors = []
    test_errors = []
    eps = config.exp_eps
    print("experiment5 flag")
    for projection in projections:

        for budget in budgets:
            dataloader=None


            for trial in range(trials):


                attacker = InfluenceAttack(method_type=args.method_type, env=env, config=config, sign=args.sign, initial_lr=args.initial_lr,
                                           type=projection, epsilon=args.epsilon,  max_epochs=args.max_epochs, reg=args.reg,eps=eps, random=1, iters=args.iters, attacker_type="influence",is_type=args.is_type,dataloader=dataloader,frac=budget, dataset_id=args.dataset_id)


                if index==0:
                    ret_policy1 = attacker.model.get_initial_return()
                    init_returns_policy1.append(ret_policy1)
                    ret_policy1 = attacker.dataLoader.eval_rets
                    init_returns_policy1.append(ret_policy1)
                    # index+=1

                attacker.attack()

                ret_policy1 = attacker.model.get_initial_return()

                init_returns_policy1.append(ret_policy1)

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                np.save("logs/experiment5_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(eps) + "_" + str(args.seed) + "_policy1.npy",
                        np.array(init_returns_policy1))


                new_test_error, _ , _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                np.save("logs/experiment5_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(eps) + "_" +str(args.seed) +"_policy1.npy",
                    np.array(init_returns_policy1))


                new_test_error, _ , _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                np.save(
                    "logs/experiment5_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps) + "_" + str(args.seed) +"_pbe.npy",
                    np.array(train_errors))
                np.save(
                    "logs/experiment5_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps) + "_" + str(args.seed) +"_train.npy",
                    np.array(test_errors))

                #model.save_model(train_errors,test_errors)
                print("pbe", pbe)
                index+=1
                attacker.dataLoader.reset()
                dataloader = attacker.dataLoader


        print(init_returns_policy1)
        print(current_budgets)


def experiment6(env,eps=0.05, trials=20):
    init_returns_policy1 = []



    init_sum=[]
    current_budgets = []
    attacker=None
    index=0
    dataloader=None
    train_errors = []
    test_errors = []
    eps = config.exp_eps
    for projection in projections:


        for budget in budgets:
            dataloader=None


            for trial in range(trials):


                attacker = InfluenceAttack(method_type=args.method_type, env=env, config=config, sign=args.sign, initial_lr=args.initial_lr,
                                           type=projection, epsilon=args.epsilon,  max_epochs=args.max_epochs, reg=args.reg,eps=eps, random=2, iters=args.iters, attacker_type="influence",is_type=args.is_type,dataloader=dataloader,frac=budget, dataset_id=args.dataset_id)


                if index==0:
                    ret_policy1 = attacker.model.get_initial_return()
                    init_returns_policy1.append(ret_policy1)
                    ret_policy1 = attacker.dataLoader.eval_rets
                    init_returns_policy1.append(ret_policy1)
                    # index+=1

                attacker.attack()

                ret_policy1 = attacker.model.get_initial_return()

                init_returns_policy1.append(ret_policy1)

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                np.save("logs/experiment6_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(eps) + "_" + str(args.seed) + "_policy1.npy",
                        np.array(init_returns_policy1))


                new_test_error, _ , _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                np.save("logs/experiment6_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(eps) + "_" +str(args.seed) +"_policy1.npy",
                        np.array(init_returns_policy1))


                new_test_error, _ , _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                np.save(
                    "logs/experiment6_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps) + "_" + str(args.seed) +"_pbe.npy",
                    np.array(train_errors))
                np.save(
                    "logs/experiment6_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps) + "_" + str(args.seed) +"_train.npy",
                    np.array(test_errors))

                #model.save_model(train_errors,test_errors)
                print("pbe", pbe)
                index+=1
                attacker.dataLoader.reset()
                dataloader = attacker.dataLoader


        print(init_returns_policy1)
        print(current_budgets)



def experiment8(env,eps=0.05, trials=1):
    # budgets=[0.5,0.4]
    init_returns_policy1 = []

    init_sum=[]
    current_budgets = []
    attacker=None
    index=0
    dataloader=None
    train_errors = []
    test_errors = []
    eps = config.exp_eps
    before_weights=None
    after_weights=None
    for projection in projections:
        dataloader=None

        for budget in budgets:

            for trial in range(trials):


                attacker = InfluenceAttack(method_type=args.method_type, env=env, config=config, sign=args.sign, initial_lr=args.initial_lr,
                                           type=projection, epsilon=args.epsilon,  max_epochs=args.max_epochs, reg=args.reg,eps=eps, random=0, iters=args.iters, attacker_type='fsgm',is_type=args.is_type,dataloader=dataloader,frac=budget, dataset_id=args.dataset_id)

                if index==0:
                    ret_policy1 = attacker.model.get_initial_return()
                    init_returns_policy1.append(ret_policy1)
                    ret_policy1 = attacker.dataLoader.eval_rets
                    init_returns_policy1.append(ret_policy1)
                    new_train_error, _, __, pbe = attacker.model.get_train_error()
                    new_test_error, _ , _ = attacker.model.get_test_error()
                    train_errors.append(new_train_error.detach().cpu().numpy())
                    test_errors.append(new_test_error.detach().cpu().numpy())
                ret_policy1 = attacker.model.get_initial_return()

                if args.method_type=="IS":
                    before_weights = attacker.model.is_weights

                attacker.attack()

                ret_policy1 = attacker.model.get_initial_return()

                if args.method_type=="IS":
                    after_weights = attacker.model.is_weights


                init_returns_policy1.append(ret_policy1)

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                np.save("logs/experiment8_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(eps) + "_" + str(args.seed) + "_policy1.npy",
                        np.array(init_returns_policy1))


                new_test_error, _ , _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                np.save(
                    "logs/experiment8_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps)  + "_"+ str(args.seed) + "_pbe.npy",
                    np.array(train_errors))
                np.save(
                    "logs/experiment8_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps)  + "_"+ str(args.seed) + "_train.npy",
                    np.array(test_errors))


                if ((budget==0.5 or budget==0.1) and (projection=="l1" and args.dataset_id in np.arange(4))):
                    attacker.save_model(np.array(train_errors),np.array(test_errors),before_weights,after_weights)
                print("pbe", pbe)
                index+=1
            dataloader = attacker.dataLoader


        print(init_returns_policy1)
        print(current_budgets)

def experiment9(env,eps=0.05, trials=1):
    # budgets=[1.0,0.4]
    init_returns_policy1 = []

    init_sum=[]
    current_budgets = []
    attacker=None
    index=0
    dataloader=None
    train_errors = []
    test_errors = []
    eps = config.exp_eps
    before_weights=None
    after_weights=None
    for projection in projections:
        dataloader=None

        for budget in budgets:

            for trial in range(trials):


                attacker = InfluenceAttack(method_type=args.method_type, env=env, config=config, sign=args.sign, initial_lr=args.initial_lr,
                                           type=projection, epsilon=args.epsilon,  max_epochs=args.max_epochs, reg=args.reg,eps=eps, random=3, iters=args.iters, attacker_type='influence',is_type=args.is_type,dataloader=dataloader,frac=budget, dataset_id=args.dataset_id)

                if index==0:
                    ret_policy1 = attacker.model.get_initial_return()
                    init_returns_policy1.append(ret_policy1)
                    ret_policy1 = attacker.dataLoader.eval_rets
                    init_returns_policy1.append(ret_policy1)
                    new_train_error, _, __, pbe = attacker.model.get_train_error()
                    new_test_error, _ , _ = attacker.model.get_test_error()
                    train_errors.append(new_train_error.detach().cpu().numpy())
                    test_errors.append(new_test_error.detach().cpu().numpy())
                ret_policy1 = attacker.model.get_initial_return()

                if args.method_type=="IS":
                    before_weights = attacker.model.is_weights

                attacker.attack()

                ret_policy1 = attacker.model.get_initial_return()

                if args.method_type=="IS":
                    after_weights = attacker.model.is_weights


                init_returns_policy1.append(ret_policy1)

                print("returns policy 1", init_returns_policy1)
                print("budget", budget)
                print("projection", projection)

                np.save("logs/experiment9_"+ attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(eps) + "_" + str(args.seed) + "_policy1.npy",
                        np.array(init_returns_policy1))


                new_test_error, _ , _ = attacker.model.get_test_error()

                # get new train error
                new_train_error, _, __, pbe = attacker.model.get_train_error()

                train_errors.append(new_train_error.detach().cpu().numpy())
                test_errors.append(new_test_error.detach().cpu().numpy())

                np.save(
                    "logs/experiment9_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps)  + "_"+ str(args.seed) + "_pbe.npy",
                    np.array(train_errors))
                np.save(
                    "logs/experiment9_" + attacker.method_type + "_" + attacker.is_type + "_" + str(attacker.dataset_id) + "_"  + attacker.attacker_type + "_" + env.name  + "_" + str(
                        eps)  + "_"+ str(args.seed) + "_train.npy",
                    np.array(test_errors))

                index+=1
            dataloader = attacker.dataLoader


        print(init_returns_policy1)
        print(current_budgets)

def reload_modules():
    import importlib
    for module in sys.modules.values():
        importlib.reload(module)

id = int(args.experiment_id)

if __name__ == '__main__':
    # id=1

    if id ==1:

        experiment1(env,trials=1)

        experiment4(env,trials=50)



        experiment1(env,trials=1)

        experiment8(env,trials=1)


        experiment2(env,trials=1)



# # # if id==1:
    # # # if id==5:
    #     experiment3(trials=1)

    elif id==2:
        experiment6(env,trials=50)
    else:
        experiment5(env,trials=100)




    # # # reload_modules()
    #
    # # reload_modules()
    # if id==2:

    # reload_modules()
    #

    # reload_modules()

    # if id==3:
    #     experiment4(env,trials=100)
    # reload_modules()
    #

    #
    #
    #
    #
