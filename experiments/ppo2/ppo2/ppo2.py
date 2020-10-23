from baselines.common import set_global_seeds
from baselines.common.policies import build_policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def predict(*, network, env, seed=None, nsteps=2048, ent_coef=0.0,
            vf_coef=0.5, max_grad_norm=0.5, nminibatches=4,
            load_path=None, model_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    if load_path is not None:
        model.load(load_path)

    if init_fn is not None:
        init_fn()

    return model
