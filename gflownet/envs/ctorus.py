"""
Classes to represent hyper-torus environments
"""
from typing import List, Tuple
import itertools
import numpy as np
import pandas as pd
import torch
from gflownet.envs.base import GFlowNetEnv
from torch.distributions import Categorical, Uniform, VonMises
from torchtyping import TensorType


class ContinuousTorus(GFlowNetEnv):
    """
    Continuous hyper-torus environment in which the action space consists of the
    increment of the angle of dimension d and the trajectory is of fixed length
    length_traj.

    The states space is the concatenation of the angle (in radians) at each dimension
    and the number of actions.

    Attributes
    ----------
    ndim : int
        Dimensionality of the torus

    length_traj : int
       Fixed length of the trajectory.
    """

    def __init__(
        self,
        n_dim=2,
        length_traj=1,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        reward_norm_std_mult=0,
        reward_func="boltzmann",
        denorm_proxy=False,
        energies_stats=None,
        proxy=None,
        oracle=None,
        **kwargs,
    ):
        super(ContinuousTorus, self).__init__(
            env_id,
            reward_beta,
            reward_norm,
            reward_norm_std_mult,
            reward_func,
            energies_stats,
            denorm_proxy,
            proxy,
            oracle,
            **kwargs,
        )
        self.continuous = True
        self.n_dim = n_dim
        self.length_traj = length_traj
        # Initialize angles and state attributes
        self.reset()
        self.source = self.angles.copy()
        self.obs_dim = self.n_dim + 1
        self.action_space = self.get_actions_space()
        self.policy_output_dim = self.get_policy_output_dim()
        self.eos = self.n_dim
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def get_actions_space(self):
        """
        Constructs list with all possible actions. The actions are tuples with two
        values: (dimension, magnitude) where dimension indicates the index of the
        dimension on which the action is to be performed and magnitude indicates the
        increment of the angle in radians.
        """
        actions = [(d, None) for d in range(self.n_dim)]
        return actions

    def get_policy_output_dim(self):
        """
        Returns the dimensionality of the output of the policy model, from which an
        action is to be determined or sampled.

        For each dimension of the hyper-torus, the output of the policy should return
        1) a logit, for the categorical distribution over dimensions and 2) the
        location and 3) the concentration of the projected normal distribution to
        sample the increment of the angle. Therefore, the output of the policy model
        has dimensionality D x 3 + 1, where D is the number of dimensions, and the elements
        of the output vector are:
        - d * 3: logit of dimension d
        - d * 3 + 1: location of Von Mises distribution for dimension d
        - d * 3 + 2: log concentration of Von Mises distribution for dimension d
        with d in [0, ..., D]
        """
        # TODO: + 1 (eos)?
        return self.n_dim * 3 + 1

    def get_mask_invalid_actions(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if action is invalid
        given the current state, False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space) + 1)]
        if state[-1] >= self.length_traj:
            mask = [True for _ in range(len(self.action_space) + 1)]
            mask[-1] = False
        else:
            mask = [False for _ in range(len(self.action_space) + 1)]
            mask[-1] = True
        return mask

    def true_density(self):
        # TODO
        # Return pre-computed true density if already stored
        if self._true_density is not None:
            return self._true_density
        # Calculate true density
        all_x = self.get_all_terminating_states()
        all_oracle = self.state2oracle(all_x)
        rewards = self.oracle(all_oracle)
        self._true_density = (
            rewards / rewards.sum(),
            rewards,
            list(map(tuple, all_x)),
        )
        return self._true_density

    def state2proxy(self, state_list: List[List]) -> List[List]:
        """
        Prepares a list of states in "GFlowNet format" for the proxy: a list of length
        n_dim with an angle in radians. The n_actions item is removed.

        Args
        ----
        state_list : list of lists
            List of states.
        """
        return [state[:-1] for state in state_list]

    def state2oracle(self, state_list: List[List]) -> List[List]:
        """
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state_list : list of lists
            List of states.
        """
        return self.state2proxy(state_list)

    def state2obs(self, state: List = None) -> List:
        """
        Maps the angles part of the state given as argument (or self.state if
        None) onto [0, 2 * pi].
        """
        if state is None:
            state = self.state.copy()
        angles = np.array(state[:-1]) % 2 * np.pi
        return angles.tolist() + [state[-1]]

    def obs2state(self, obs: List) -> List:
        """
        Simply returns the input as is.
        """
        return obs

    def state2readable(self, state: List) -> str:
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state. Angles are converted into degrees in [0, 360]
        """
        angles = np.array(state[:-1]) % 2 * np.pi
        angles = angles * 180 / np.pi
        angles = str(angles).replace("(", "[").replace(")", "]").replace(",", "")
        n_actions = str(state[-1])
        return angles + " | " + n_actions

    def readable2state(self, readable: str) -> List:
        """
        Converts a human-readable string representing a state into a state as a list of
        positions. Angles are converted back to radians.
        """
        pair = readable.split(" | ")
        angles = [np.float32(el) * np.pi / 180 for el in pair[0].strip("[]").split(" ")]
        n_actions = [int(pair[1])]
        return angles + n_actions

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.angles = [0.0 for _ in range(self.n_dim)]
        # TODO: do step encoding as in Sasha's code?
        self.n_actions = 0
        # States are the concatenation of the angle state and number of actions
        self.state = self.angles + [self.n_actions]
        self.done = False
        self.id = env_id
        return self

    # TODO: we might have to do something with the action number
    def get_parents(self, state=None, done=None):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length n_angles where each element
            is the position at each dimension.

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents as state2obs(state)

        actions : list
            List of actions that lead to state for each parent in parents
        """

        def _get_min_actions_to_source(source, ref):
            def _get_min_actions_dim(u, v):
                return np.min([np.abs(u - v), np.abs(u - (v - self.n_angles))])

            return np.sum([_get_min_actions_dim(u, v) for u, v in zip(source, ref)])

        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [self.state2obs(state)], [self.eos]
        # If source state
        elif state[-1] == 0:
            return [], []
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                state_p = state.copy()
                angles_p = state_p[: self.n_dim]
                n_actions_p = state_p[-1]
                # Get parent
                n_actions_p -= 1
                if a[0] != -1:
                    angles_p[a[0]] -= a[1]
                    # If negative angle index, restart from the back
                    if angles_p[a[0]] < 0:
                        angles_p[a[0]] = self.n_angles + angles_p[a[0]]
                    # If angle index larger than n_angles, restart from 0
                    if angles_p[a[0]] >= self.n_angles:
                        angles_p[a[0]] = angles_p[a[0]] - self.n_angles
                if _get_min_actions_to_source(self.source, angles_p) < state[-1]:
                    state_p = angles_p + [n_actions_p]
                    parents.append(self.state2obs(state_p))
                    actions.append(idx)
        return parents, actions

    def sample_actions(
        self,
        policy_outputs: TensorType["batch_size", "policy_output_dim"],
        sampling_method: str = "policy",
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
        temperature_logits: float = 1.0,
        loginf: float = 1000,
    ) -> Tuple[List[Tuple], TensorType["batch_size"]]:
        """
        Samples a batch of actions from a batch of policy outputs.
        """
        batch_size = policy_outputs.shape[0]
        bs_range = torch.arange(batch_size)
        # Sample dimensions
        if sampling_method == "uniform":
            logits_dims = torch.zeros(batch_size, self.n_dim).to(policy_outputs)
        elif sampling_method == "policy":
            logits_dims = policy_outputs[:, 0 :: 3]
            logits_dims /= temperature_logits
        if mask_invalid_actions is not None:
            logits_dims[mask_invalid_actions] = -loginf
        dimensions = Categorical(logits=logits_dims).sample()
        logprobs_dim = self.logsoftmax(logits_dims)[bs_range, dimensions]
        # Sample angle increments
        if sampling_method == "uniform":
            distr_angles = Uniform(
                torch.zeros(batch_size), 2 * torch.pi * torch.ones(batch_size)
            )
        elif sampling_method == "policy":
            locations = policy_outputs[:, 1 :: 3][bs_range, dimensions]
            concentrations = policy_outputs[:, 2 :: 3][bs_range, dimensions]
            distr_angles = VonMises(locations, torch.exp(concentrations))
        angles = distr_angles.sample()
        logprobs_angles = distr_angles.log_prob(angles)
        # Combined probabilities
        logprobs = logprobs_dim + logprobs_angles
        # Build actions
        actions = [(dimension, angle) for dimension, angle in zip(dimensions.tolist(), angles.tolist())]
        return actions, logprobs

    def step(self, action):
        """
        Executes step given an action.

        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action_idx : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if self.done:
            return self.state, action_idx, False
        # If only possible action is eos, then force eos
        # If the number of actions is equal to maximum trajectory length
        elif self.n_actions == self.length_traj:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        elif action_idx != self.eos:
            a_dim, a_dir = self.action_space[action_idx]
            angles_next = self.angles.copy()
            if a_dim != -1:
                angles_next[a_dim] += a_dir
                # If negative angle index, restart from the back
                if angles_next[a_dim] < 0:
                    angles_next[a_dim] = self.n_angles + angles_next[a_dim]
                # If angle index larger than n_angles, restart from 0
                if angles_next[a_dim] >= self.n_angles:
                    angles_next[a_dim] = angles_next[a_dim] - self.n_angles
            self.angles = angles_next
            self.n_actions += 1
            self.state = self.angles + [self.n_actions]
            valid = True
            return self.state, action_idx, valid
        # If action is eos, then it is invalid
        else:
            return self.state, self.eos, False

    def make_train_set(self, ntrain, oracle=None, seed=168, output_csv=None):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        """
        rng = np.random.default_rng(seed)
        angles = rng.integers(low=0, high=self.n_angles, size=(ntrain,) + (self.n_dim,))
        n_actions = self.length_traj * np.ones([ntrain, 1], dtype=np.int32)
        samples = np.concatenate([angles, n_actions], axis=1)
        if oracle:
            energies = oracle(self.state2oracle(samples))
        else:
            energies = self.oracle(self.state2oracle(samples))
        df_train = pd.DataFrame({"samples": list(samples), "energies": energies})
        if output_csv:
            df_train.to_csv(output_csv)
        return df_train

    def make_test_set(self, config):
        """
        Constructs a test set.

        Args
        ----
        """
        if "all" in config and config.all:
            samples = self.get_all_terminating_states()
            energies = self.oracle(self.state2oracle(samples))
        df_test = pd.DataFrame(
            {"samples": [self.state2readable(s) for s in samples], "energies": energies}
        )
        return df_test

    def get_all_terminating_states(self):
        all_x = np.int32(
            list(itertools.product(*[list(range(self.n_angles))] * self.n_dim))
        )
        n_actions = self.length_traj * np.ones([all_x.shape[0], 1], dtype=np.int32)
        all_x = np.concatenate([all_x, n_actions], axis=1)
        return all_x
