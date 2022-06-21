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
#import gurobipy as gp
# from gurobipy import GRB
sys.path.append("domains_for_bilevel/interpretable_ope_public/")

sys.path.append("domains_for_bilevel/interpretable_ope_public/Cancer/")

def l2_projection_custom(next_states, actual_next_states, epsilon):
    """
        Computes l_{2} projection without gurobi
        :param next_states: features to be projected
        :param actual_next_states: center of l_{2} ball
        :param epsilon: radius of the l_{2} ball
        :return:
        """
    if np.all(np.sqrt(np.sum(np.square(actual_next_states - next_states),axis=1)) <= epsilon)is True:
        return next_states

    # scale so that we can project on zero-center l_2 norm ball
    y = next_states - actual_next_states
    # x = min{1, radius/|y|} *y closed form of l2 projection
    x = epsilon/np.maximum(epsilon,np.linalg.norm(y,ord=2,axis=1)+1e-10).reshape(-1,1)*y
    # rescale so that the projection is on l_2 norm ball centered at actual_next_states
    scaled_x = x + actual_next_states
    return scaled_x


def l2_projection(next_states, actual_next_states, epsilon):
    """
        Computes l_{2} projection with gurobi
        :param next_states: features to be projected
        :param actual_next_states: center of l_{1} ball
        :param epsilon: radius of the l_{1} ball
        :return:
        """
    model = gp.Model("bilinear")
    # model.params.NonConvex = 2

    n = next_states.shape[0]
    dim = next_states.shape[1]
    print("shape", actual_next_states.shape)

    phi_ = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights", lb=-GRB.INFINITY)

    auxilary = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights2", lb=-GRB.INFINITY)
    auxilary_new = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights5",
                                 lb=-GRB.INFINITY)

    model.setObjective(
        gp.quicksum(auxilary_new[idx] * auxilary_new[idx] for idx in range(n * dim)),
        GRB.MINIMIZE)

    for idx in range(n):

        for jdx in range(dim):
            model.addConstr(
                auxilary[idx * dim + jdx] == actual_next_states[idx][jdx] - phi_[
                    idx * dim + jdx],
                "_" + str("third_constr") + str(jdx) + str(idx))

            model.addConstr(
                auxilary_new[idx * dim + jdx] == next_states[idx][jdx] - phi_[
                    idx * dim + jdx],
                "_" + str("fifth_constr") + str(jdx) + str(idx))


        model.addConstr(gp.quicksum(
            auxilary[idx * dim + jdx] * auxilary[idx * dim + jdx] for jdx in range(dim))
        , GRB.LESS_EQUAL, epsilon,
        "_" + str("second_constr"))

    model.setParam('OutputFlag', False)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        next_states_corrupt = model.getAttr('x', phi_)

    if model.status == GRB.INFEASIBLE:
        print("infeasible")
    print('Obj: %s' % model.ObjVal)

    print("weights")
    print("l2 diff", np.sum(np.abs(np.array(next_states_corrupt.values()).reshape((n, dim))-actual_next_states)**2))

    return np.array(next_states_corrupt.values()).reshape((n, dim)), model.ObjVal


def l1_projection(next_states, actual_next_states, epsilon):
    """
        Computes l_{1} projection without gurobi
        :param next_states: features to be projected
        :param actual_next_states: center of l_{1} ball
        :param epsilon: radius of the l_{1} ball
        :return:
        """
    model = gp.Model("bilinear")
    # model.params.NonConvex = 2

    n = next_states.shape[0]
    dim = next_states.shape[1]
    model = gp.Model("bilinear")
    # model.params.NonConvex = 2
    ###### corrupt next state features ################
    phi_ = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights", lb=-GRB.INFINITY)

    ############# difference between actual next state features - corrupt next state features ########
    auxilary = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights2", lb=-GRB.INFINITY)

    ############# absolute difference between actual next state features - corrupt next state features ########
    auxilary_abs = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights1",
                                 lb=-GRB.INFINITY)

    ############# difference between next state features derived using influence attack - corrupt next state features ########
    auxilary_new = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights5",
                                 lb=-GRB.INFINITY)

    ############# difference between next state features derived using influence attack - corrupt next state features ########
    auxilary_new_abs = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights1",
                                     lb=-GRB.INFINITY)

    ########### minimizes distance between corrupt next state features derived using influence and corrupt next state features ########
    model.setObjective(
        gp.quicksum(auxilary_new_abs[idx]*auxilary_new_abs[idx] for idx in range(n * dim)),
        GRB.MINIMIZE)

    ######## creates auxilary variables and auxiliary_abs variables ############
    for idx in range(n):

        for jdx in range(dim):
            model.addConstr(auxilary[idx * dim + jdx] == actual_next_states[idx][jdx] - phi_[
                idx * dim + jdx],
                            "_" + str("third_constr") + str(jdx) + str(idx))
            model.addConstr(auxilary_abs[idx * dim + jdx] == gp.abs_(auxilary[idx * dim + jdx]),
                            "_" + str("fourth_constr") + str(jdx) + str(idx))

        ######## creates auxilary new variables and auxiliary_new_abs variables ############
        for jdx in range(dim):
            model.addConstr(auxilary_new[idx * dim + jdx] == next_states[idx][jdx] - phi_[
                idx * dim + jdx],
                            "_" + str("fifth_constr") + str(jdx) + str(idx))
            model.addConstr(
                auxilary_new_abs[idx * dim + jdx] == gp.abs_(auxilary_new[idx * dim + jdx]),
                "_" + str("sixth_constr") + str(jdx) + str(idx))

        ############ Constraint to make sure that L1 distance between actual next state features and corrupt next state features is not above epsilon#######
        model.addConstr(
        gp.quicksum(auxilary_abs[idx * dim + jdx] for jdx in range(dim))
        , GRB.LESS_EQUAL, epsilon,
        "_" + str("second_constr"))

    model.setParam('OutputFlag', False)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        next_states_corrupt = model.getAttr('x', phi_)

    if model.status == GRB.INFEASIBLE:
        print("infeasible")
    print('Obj: %s' % model.ObjVal)

    print("weights")

    return np.array(next_states_corrupt.values()).reshape((n, dim)), model.ObjVal


def linf_projection_custom(next_states, actual_next_states, epsilon):
    """
    Computes l_{infinity} projection without gurobi
    :param next_states: features to be projected
    :param actual_next_states: center of l_{inf} ball
    :param epsilon: radius of the l_{inf} ball
    :return:
    """
    # clip next_state features that lie outside of actual_next_states - r < x < actual_next_states +r
    projected_corrupt_xtrain = np.clip(next_states.flatten(), a_min=actual_next_states.flatten() - epsilon,
                                       a_max=actual_next_states.flatten() + epsilon).reshape(next_states.shape)
    return projected_corrupt_xtrain


def linf_projection(next_states, actual_next_states, epsilon):
    """

    :param next_states:
    :param actual_next_states:
    :param epsilon:
    :return:
    """
    model = gp.Model("bilinear")
    # model.params.NonConvex = 2

    n = next_states.shape[0]
    dim = next_states.shape[1]
    model = gp.Model("bilinear")
    # model.params.NonConvex = 2
    ###### corrupt next state features ################
    phi_ = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights", lb=-GRB.INFINITY)

    ############# difference between actual next state features - corrupt next state features ########
    auxilary = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights2", lb=-GRB.INFINITY)

    ############# absolute difference between actual next state features - corrupt next state features ########
    auxilary_abs = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights1",
                                 lb=-GRB.INFINITY)

    ############# difference between next state features derived using influence attack - corrupt next state features ########
    auxilary_new = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights5",
                                 lb=-GRB.INFINITY)

    ############# difference between next state features derived using influence attack - corrupt next state features ########
    auxilary_new_abs = model.addVars(range(n * dim), vtype=GRB.CONTINUOUS, name="weights1",
                                     lb=-GRB.INFINITY)


    t = model.addVar(lb=-GRB.INFINITY,name="t")

    t_new = model.addVar(lb=-GRB.INFINITY,name="t_new")



    ########### minimizes distance between corrupt next state features derived using influence and corrupt next state features ########
    model.setObjective(
       t_new,
        GRB.MINIMIZE)

    ######## creates auxilary variables and auxiliary_abs variables ############
    for idx in range(n):

        for jdx in range(dim):
            model.addConstr(auxilary[idx * dim + jdx] == actual_next_states[idx][jdx] - phi_[
                idx * dim + jdx],
                            "_" + str("third_constr") + str(jdx) + str(idx))
            model.addConstr(auxilary_abs[idx * dim + jdx] == gp.abs_(auxilary[idx * dim + jdx]),
                            "_" + str("fourth_constr") + str(jdx) + str(idx))

            model.addConstr(auxilary_abs[idx * dim + jdx], GRB.LESS_EQUAL, t,
                            "_" + str("t_2_constr") + str(jdx) + str(idx))

        ######## creates auxilary new variables and auxiliary_new_abs variables ############
        for jdx in range(dim):
            model.addConstr(auxilary_new[idx * dim + jdx] == next_states[idx][jdx] - phi_[
                idx * dim + jdx],
                            "_" + str("fifth_constr") + str(jdx) + str(idx))
            model.addConstr(
                auxilary_new_abs[idx * dim + jdx] == gp.abs_(auxilary_new[idx * dim + jdx]),
                "_" + str("sixth_constr") + str(jdx) + str(idx))

            model.addConstr(auxilary_new_abs[idx * dim + jdx], GRB.LESS_EQUAL, t_new,
                            "_" + str("t_constr") + str(jdx) + str(idx))

            model.addConstr(t, GRB.LESS_EQUAL, epsilon,
                            "_" + str("t_1_constr") + str(jdx) + str(idx))

    model.setParam('OutputFlag', False)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        next_states_corrupt = model.getAttr('x', phi_)

    print(np.max(np.abs(next_states.flatten()-next_states_corrupt.values())))
    print(np.max(np.abs(actual_next_states.flatten()-next_states_corrupt.values())))


    if model.status == GRB.INFEASIBLE:
        print("infeasible")
    print('Obj: %s' % model.ObjVal)

    print("weights")

    return np.array(next_states_corrupt.values()).reshape((n, dim)), model.ObjVal

# def euclidean_proj_simplex(v, s=1):
#     """ Compute the Euclidean projection on a positive simplex
#     Solves the optimisation problem (using the algorithm from [1]):
#         min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
#     Parameters
#     ----------
#     v: (n,) numpy array,
#        n-dimensional vector to project
#     s: int, optional, default: 1,
#        radius of the simplex
#     Returns
#     -------
#     w: (n,) numpy array,
#        Euclidean projection of v on the simplex
#     Notes
#     -----
#     The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
#     Better alternatives exist for high-dimensional sparse vectors (cf. [1])
#     However, this implementation still easily scales to millions of dimensions.
#     References
#     ----------
#     [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
#         John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
#         International Conference on Machine Learning (ICML 2008)
#         http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
#         https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
#     """
#     assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
#     m = v.shape[0]
#     n = v.shape[1]  # will raise ValueError if v is not 1-D
#     # check if we are already on the simplex
#     if np.sum(np.sum(v,axis=-1) == s)==m and np.alltrue(v >= 0):
#         # best projection: itself!
#         return v
#     # get the array of cumulative sums of a sorted (decreasing) copy of v
#
#     u = np.sort(v,axis=1)[:,::-1]
#     cssv = np.cumsum(u,axis=-1)
#     # get the number of > 0 components of the optimal solution
#     rho = [ np.nonzero(u[idx,:] * np.arange(1, n+1) > (cssv[idx,:] - s))[0][-1] for idx in range(m)]
#     rho = np.array(rho).flatten()
#     # compute the Lagrange multiplier associated to the simplex constraint
#     theta = (cssv[np.arange(rho.shape[0]),rho] - s) / (rho + 1e-20)
#     # compute the projection by thresholding v using theta
#     w = (v - theta.reshape(-1,1)).clip(min=0)
#     return w
#
#
# def euclidean_proj_l1ball(v, s=1):
#     """ Compute the Euclidean projection on a L1-ball
#     Solves the optimisation problem (using the algorithm from [1]):
#         min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
#     Parameters
#     ----------
#     v: (n,) numpy array,
#        n-dimensional vector to project
#     s: int, optional, default: 1,
#        radius of the L1-ball
#     Returns
#     -------
#     w: (n,) numpy array,
#        Euclidean projection of v on the L1-ball of radius s
#     Notes
#     -----
#     Solves the problem by a reduction to the positive simplex case
#     See also
#     --------
#     euclidean_proj_simplex
#     https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
#     """
#     assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
#     m = v.shape[0]
#     n = v.shape[1]  # will raise ValueError if v is not 1-D
#     # compute the vector of absolute values
#     u = np.abs(v)
#     # check if v is already a solution
#     if np.sum(np.sum(u,axis=1) <= s)==m:
#         # L1-norm is <= s
#         return v
#     # v is not already a solution: optimum lies on the boundary (norm == s)
#     # project *u* on the simplex
#     w = euclidean_proj_simplex(u, s=s)
#     # compute the solution to the original problem on v
#     w *= np.sign(v)
#     return w

def euclidean_proj_simplex(v, s=1):

    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

    https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n = v.shape[1]  # will raise ValueError if v is not 1-D
    m = v.shape[0]
    # check if we are already on the simplex
    if np.sum(np.sum(v,-1) == s)==m and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v,axis=1)[:,::-1]
    cssv = np.cumsum(u,axis=1)
    # get the number of > 0 components of the optimal solution
    #rho_vals = [ u[idx,:] * np.arange(1, n+1) > (cssv[idx,:] - s) for idx in range(m)]

    #rho = [np.nonzero(u[idx,:] * np.arange(1, n+1) > (cssv[idx,:] - s))[0][-1] for idx in range(m)]
    #rho = np.array(rho).flatten()
    #rho = []
    w=[]
    for idx in range(m):
        k = np.nonzero(u[idx, :] * np.arange(1, n + 1) > (cssv[idx, :] - s))[0]
        if len(k)==0:
            w.append(v[idx,:])
        else:
            rho = k[-1]
            theta = ((cssv[idx,rho] - s)*1.0)/(rho+1.0)
            w.append((v[idx,:] - theta).clip(min=0))
    return np.array(w)


    # compute the Lagrange multiplier associated to the simplex constraint
    #theta = ((cssv[np.arange(m),rho] - s)*1.0)/ (rho.flatten()+1)
    # compute the projection by thresholding v using theta
    #w = (v - theta.reshape(-1,1)).clip(min=0)


    #return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n = v.shape[1]  # will raise ValueError if v is not 1-D
    m = v.shape[0]
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if np.sum(np.sum(u,axis=1) <= s) == m:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w

def l1_projection_custom(next_states, actual_next_states, epsilon):
    """
    Computes projection onto L_1 norm ball with center = actual_next_states and radius=epsilon according to
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    :param next_states: features to be projected
    :param actual_next_states: center of l_{1} ball
    :param epsilon: radius of the l_{1} ball
    :return:
    Taken from https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
    """
    # rescale so that the projection is centered on l_0
    shape = next_states.shape
    # next_states  = next_states.flatten()
    # actual_next_states = actual_next_states.flatten()

    y = next_states - actual_next_states
    s = epsilon


    projected = euclidean_proj_l1ball(y,s=s)
    projected_next_states = projected + actual_next_states
    return projected_next_states.reshape(shape)





def get_projected_features(corrupt_data, orig_data, epsilon, type):

     """
     Projects corrupt features on a norm ball centered around original features
     :param self:
     :param corrupt_data:
     :param orig_data:
     :param epsilon:
     :param type:
     :return:
     """
     if epsilon == 0:
         return orig_data

     if isinstance(corrupt_data,np.ndarray)==False:
         corrupt_data = corrupt_data.detach().cpu().numpy()

     if isinstance(orig_data, np.ndarray) == False:
         orig_data = orig_data.detach().cpu().numpy()


     cor_budget = epsilon +1

     if type == "l2":

            projected_corrupt_xtrain_custom = l2_projection_custom(corrupt_data, orig_data,
                                                                   epsilon=epsilon)

            # projected_corrupt_xtrain, _ = l2_projection(corrupt_data, orig_data, epsilon=self.epsilon)
            projected_corrupt_xtrain = projected_corrupt_xtrain_custom


            """ Asserts that the projected corrupt data does not violate the attacker's budget constraints """
            # assert (np.all(np.sqrt(np.sum(np.square(orig_data - projected_corrupt_xtrain),axis=1)) <= cor_budget))
            print("state diff l2", np.sum(np.sqrt(np.sum(np.square(orig_data - projected_corrupt_xtrain),axis=1))))
            print(np.sqrt(epsilon))

     elif type == "l1":
            projected_corrupt_xtrain_custom = l1_projection_custom(corrupt_data, orig_data, epsilon=epsilon)
            # projected_corrupt_xtrain, _ = l1_projection(corrupt_data, orig_data, epsilon=self.epsilon)
            # print(np.sum(np.abs(projected_corrupt_xtrain_custom-projected_corrupt_xtrain)))
            projected_corrupt_xtrain = projected_corrupt_xtrain_custom

            # assert(np.sum(np.abs(projected_corrupt_xtrain_custom-projected_corrupt_xtrain))<=1.0)

            """ Asserts that the projected corrupt data does not violate the attacker's budget constraints """
            print(np.sum(np.sum(np.abs(orig_data - projected_corrupt_xtrain),axis=1)))
            # assert (np.all(np.sum(np.abs(orig_data - projected_corrupt_xtrain),axis=1) <= cor_budget))
            print("state diff l1", np.sum(np.abs(orig_data - projected_corrupt_xtrain)))
     else:
            print("priginal data shape", orig_data.shape)
            #scaling linf projection
            #epsilon = epsilon/orig_data.shape[1]
            projected_corrupt_xtrain_custom = linf_projection_custom(corrupt_data, orig_data, epsilon=epsilon)
            # projected_corrupt_xtrain, _ = linf_projection(corrupt_data, orig_data, epsilon=self.epsilon)
            projected_corrupt_xtrain = projected_corrupt_xtrain_custom

            print(np.sum(np.abs(projected_corrupt_xtrain_custom - projected_corrupt_xtrain)))
            print("state diff linf", np.max(np.abs(orig_data - projected_corrupt_xtrain)))

            """ Asserts that the projected corrupt data does not violate the attacker's budget constraints """
            # assert (np.all(np.max(np.abs(orig_data - projected_corrupt_xtrain),axis=1) <= epsilon))

     return projected_corrupt_xtrain


def get_frank_wolfe_projection(influence, epsilon, type):
    """
    Projects corrupt features on a norm ball centered around original features
    :param self:
    :param corrupt_data:
    :param orig_data:
    :param epsilon:
    :param type:
    :return:
    """
    if epsilon == 0:
        return np.zeros_like(influence)

    if isinstance(influence, np.ndarray) == False:
        influence = influence.detach().cpu().numpy()

    if type == "l2":
        delta =(epsilon/np.maximum(epsilon, np.linalg.norm(influence,ord=2,axis=1))+1e-10).reshape(-1,1)*influence
        # assert(np.all(np.sqrt(np.sum(np.square(delta),axis=1))<=epsilon)==1)

    elif type == "l1":
        indices = np.argmax(influence,axis=1).astype(int)
        delta = np.zeros_like(influence)
        delta[np.arange(influence.shape[0]),indices.flatten()]=epsilon
        delta = delta * np.sign(influence)
        assert(np.all(np.sum(np.abs(delta),axis=1)<=epsilon)==1)


    else:
        sign = np.sign(influence)
        delta = np.ones(influence.shape)
        delta = delta * sign * epsilon
        assert(np.all(np.max(np.abs(delta),axis=1)<=epsilon)==1)

    return delta



def euclidean_proj_simplex_(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / rho
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball_(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex_(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w



