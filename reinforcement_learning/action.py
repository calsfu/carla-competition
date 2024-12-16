import random
import torch


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """
    actions = policy_net(torch.tensor(state, dtype=torch.float32).to('cuda')).detach()
    return actions.argmax().item()
    # TODO: Select greedy action

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """
    if random.random() < exploration.value(t):
        return random.randint(0, action_size - 1)
    else:
        return select_greedy_action(state, policy_net, action_size)
    # TODO: Select exploratory action

def get_action_set():
    """ Get the list of available actions
    Returns [steering, throttle, braking].
    -------
    list
        list of available actions
    """
    return [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 0]]
