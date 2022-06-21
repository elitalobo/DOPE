

class cartpole_config():
    # domain parameters
    state_dim = 4
    normalized =False
    transformed=True
    sign=1
    threshold=2000
    is_threshold =2000000
    flag=False
    num_trajectories= 1000 #40
    normalize_rewards = False

    policy_feature_type='rbf'
    policy_num_centroids=20


    action_size = 2
    gamma = 0.98
    n_win_ticks = 195
    max_length = 100
    oracle_reward = 1
    num_traj =1000 #30
    exp_eps = 0.05
    exp_budget = 1.0

    initial_lr = 1.0
    num_centroids = 30
    feature_type = 'rbf'
    deg = 4
    num_samples = 1000000
    reg=1e-2


    min_episodes = 10000
    expected_scores = 195
    scores_length = 100
    dp_scores_length = 30
    guide_length = -1

    rescale = [[1, 1, 1, 1]]

    # q training parameters
    dqn_batch_size = 64
    dqn_hidden_dims = [100,24]
    dqn_num_episodes = 20000
    buffer_capacity = 10000
    dqn_alpha = 0.01
    dqn_alpha_decay = 0.01
    dqn_epsilon = 1.0
    dqn_epsilon_min = 0.01
    dqn_epsilon_decay = 0.995
    sample_capacity = 200000

    # model parameters
    sample_num_traj = 1024
    train_num_traj = 900
    dev_num_traj = 124
    transition_input_dims = 4
    rep_hidden_dims = [16] # The last dim is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []
    terminal_hidden_dims = [100,32]
    behavior_epsilon = 0.2
    eval_epsilon = 0.0

    # model training parameter
    print_per_epi = 10
    train_num_episodes = 100
    train_num_batches = 100
    train_batch_size = 64
    test_batch_size = 16
    tc_num_episode = 100
    tc_num_batches = 100
    tc_batch_size = 64
    tc_test_batch_size = 16
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.01
    weight_decay = 0

    # MRDR parameter
    soften_epsilon = 0.02
    mrdr_lr = 0.01
    mrdr_num_episodes = 100
    mrdr_num_batches = 100
    mrdr_batch_size = 1000
    mrdr_test_batch_size = 100
    mrdr_hidden_dims = [32]

    eval_num_traj = 1000
    eval_num_rollout = 1
    N = 100
    MAX_SEED = 1000000



class mountaincar_config():
    # domain parameters
    state_dim = 2
    action_size = 3
    sign =1
    threshold=2000
    is_threshold=2000000
    flag=False

    normalized =False
    transformed=True
    num_trajectories= 250#40

    normalize_rewards = False

    gamma = 0.99
    max_length = 150
    oracle_reward = -1
    rescale = [[1, 10]]
    num_traj = 500
    low = [-1.2, -0.06]
    high =  [0.6 ,0.07]

    initial_lr = 1.0
    num_centroids = 50 #20
    feature_type = 'rbf'
    deg = 2
    num_samples = 2000000
    reg=1e-2

    min_episodes=20000
    expected_scores=-160
    scores_length = 100
    dp_scores_length = 100
    guide_length=10000
    exp_eps = 0.1
    exp_budget = 1.0


    # q training parameters
    dqn_batch_size = 256
    dqn_hidden_dims = [60]
    dqn_num_episodes = 30000
    buffer_capacity = 20000
    dqn_alpha = 0.01
    dqn_epsilon = 0.5
    dqn_epsilon_min = 0.01
    dqn_epsilon_decay = 0.9995
    sample_capacity = 200000
    target_update = 10

    # model parameters
    sample_num_traj = 1024
    train_num_traj = 900
    dev_num_traj = 124
    transition_input_dims = 4
    rep_hidden_dims = [16] # The last dim is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []
    terminal_hidden_dims = [32,32]
    behavior_epsilon = 0.2
    eval_epsilon = 0.0

    # model training parameter
    print_per_epi = 10
    train_num_episodes = 100
    train_num_batches = 50
    train_batch_size = 16
    test_batch_size = 16
    tc_num_episode = 100
    tc_num_batches = 50
    tc_batch_size = 16
    tc_test_batch_size = 16
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.1
    weight_decay = 0.00005

    # MRDR parameter
    soften_epsilon = 0.02
    mrdr_lr = 0.01
    mrdr_num_episodes = 100
    mrdr_num_batches = 50
    mrdr_batch_size = 1000
    mrdr_test_batch_size = 100
    mrdr_hidden_dims = [32]

    eval_num_traj = 1000
    eval_num_rollout = 1
    N = 100
    MAX_SEED = 1000000






class hiv_config():
    policy_feature_type = 'rbf'
    policy_num_centroids = 50
    sign = -1
    threshold=2000
    is_threshold=2000000
    flag=False

    normalized =False
    transformed=False

    # domain parameters
    state_dim = 6
    num_trajectories=1000 #80

    normalize_rewards = True

    action_size = 4
    gamma = 0.98
    max_length = 50
    num_traj =1000
    initial_lr = 1.0
    num_centroids = 40 #40
    feature_type = 'rbf'
    deg = 3
    num_samples = 10000000
    reg=1e-2
    exp_eps = 0.05
    exp_budget = 1.0



    # model parameters
    sample_num_traj = 40
    train_num_traj = 45
    dev_num_traj = 5
    rep_hidden_dims = [256, 256, 28] # The last layer is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []

    print_per_epi = 10
    train_num_episodes = 100
    train_num_batches = 100
    train_batch_size = 40
    test_batch_size = 5
    train_traj_batch_size = 4
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.1

    # eval_num_traj = 1000
    eval_num_rollout = 1
    eval_pib_num_rollout = 100

    N = 10

    fix_data = False

    behavior_eps = 0.05
    standardize_rewards = True
    ins = 20

    min_episodes = 50000
    expected_scores = 2674500108.34561
    scores_length = 100
    dp_scores_length = 100
    guide_length = 10000

    rescale = [[1, 1, 1, 1]]

    # q training parameters
    dqn_batch_size = 256
    # dqn_hidden_dims = [350,30]
    dqn_hidden_dims = [300,50]
    dqn_num_episodes = 200000
    buffer_capacity = 20000
    dqn_alpha = 0.01
    dqn_alpha_decay = 0.01
    dqn_epsilon = 1.0
    dqn_epsilon_min = 0.01
    dqn_epsilon_decay = 0.995
    sample_capacity = 200000


class cancer_config():

    # domain parameters
    normalized =False
    transformed=False
    state_dim = 4
    num_trajectories= 500 #500

    sign = 1
    threshold=2000
    is_threshold=2000000
    flag=False
    normalize_rewards = False
    action_size = 2
    gamma = 0.95
    max_length = 30
    num_traj = 1000 #30
    initial_lr=1.0
    num_centroids=200
    feature_type='rbf'
    deg = 4
    num_samples = 1000000
    reg=1e-2
    exp_eps = 0.05
    exp_budget = 1.0


    # model parameters
    sample_num_traj = 30
    train_num_traj = 30
    dev_num_traj = 30
    rep_hidden_dims = [64, 48] # The last layer is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []

    train_num_episodes = 100
    train_num_batches = 100
    train_batch_size = 64
    test_batch_size = 16
    tc_num_episode = 100
    tc_num_batches = 100
    tc_batch_size = 64
    tc_test_batch_size = 16
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.01
    weight_decay = 0

    # eval_num_traj = 1000
    eval_num_rollout = 1
    eval_pib_num_rollout = 100

    N = 10

    fix_data = False

    behavior_eps = 0.05
    standardize_rewards = True
    ins = 20

    min_episodes = 5000
    expected_scores =30
    scores_length = 100
    dp_scores_length = 100
    guide_length = -1

    rescale = [[1, 1, 1, 1]]

    # q training parameters
    dqn_batch_size = 400
    dqn_hidden_dims = [64, 28]
    dqn_num_episodes = 2000
    buffer_capacity = 10000
    dqn_alpha = 0.01
    dqn_alpha_decay = 0.01
    dqn_epsilon = 1.0
    dqn_epsilon_min = 0.01
    dqn_epsilon_decay = 0.995
    sample_capacity = 200000


class sepsis_config():

    # domain parameters
    state_dim = 24
    normalize_rewards = False
    action_size = 46
    gamma = 0.95
    max_length = 30
    threshold=2000
    is_threshold=2000000
    num_trajectories=100

    num_traj = 50

    initial_lr=0.01
    num_centroids=52
    feature_type='poly'
    deg = 1
    num_samples = 4000
    reg=1e-2



    # model parameters
    sample_num_traj = 40
    train_num_traj = 45
    dev_num_traj = 5
    rep_hidden_dims = [64, 64] # The last layer is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []

    print_per_epi = 10
    train_num_episodes = 100
    train_num_batches = 100
    train_batch_size = 40
    test_batch_size = 5
    train_traj_batch_size = 4
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.1

    # eval_num_traj = 1000
    eval_num_rollout = 1
    eval_pib_num_rollout = 100

    N = 10

    fix_data = False

    behavior_eps = 0.05
    standardize_rewards = True
    ins = 20


class custom_config():
    # domain parameters
    normalized =False
    transformed=False
    state_dim = 2
    num_trajectories= 500 #500

    sign = -1
    threshold=5000
    is_threshold=2000000
    flag=True



    normalize_rewards = False
    action_size = 2
    gamma = 0.95
    max_length = 50
    num_traj = 500 #30
    initial_lr=1.0
    num_centroids=200
    feature_type='rbf'
    deg = 4
    num_samples = 1000000
    reg=1e-2
    exp_eps = 0.05
    exp_budget = 1.0


    # model parameters
    sample_num_traj = 30
    train_num_traj = 30
    dev_num_traj = 30
    rep_hidden_dims = [24, 24] # The last layer is the representation dim
    transition_hidden_dims = []
    reward_hidden_dims = []

    train_num_episodes = 100
    train_num_batches = 100
    train_batch_size = 64
    test_batch_size = 16
    tc_num_episode = 100
    tc_num_batches = 100
    tc_batch_size = 64
    tc_test_batch_size = 16
    lr = 0.01
    lr_decay = 0.9
    alpha_rep = 0.01
    weight_decay = 0

    # eval_num_traj = 1000
    eval_num_rollout = 1
    eval_pib_num_rollout = 100

    N = 10

    fix_data = False

    behavior_eps = 0.05
    standardize_rewards = True
    ins = 20

    min_episodes = 5000
    expected_scores =500
    scores_length = 100
    dp_scores_length = 100
    guide_length = -1

    rescale = [[1, 1, 1, 1]]

    # q training parameters
    dqn_batch_size = 400
    dqn_hidden_dims = [24]
    dqn_num_episodes = 5000
    buffer_capacity = 10000
    dqn_alpha = 0.01
    dqn_alpha_decay = 0.01
    dqn_epsilon = 1.0
    dqn_epsilon_min = 0.01
    dqn_epsilon_decay = 0.995
    sample_capacity = 200000

