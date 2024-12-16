import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple

def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, double_dqn=False):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """
    # Referenced from here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # 1. sample from replay buffer
    # data = (obs_t, action, reward, obs_tp1, done)    
    transitions = replay_buffer.sample(batch_size)
    state_batch = torch.from_numpy(transitions[0]).to(device)
    action_batch = torch.from_numpy(transitions[1]).to(device)
    reward_batch = torch.from_numpy(transitions[2]).to(device)
    next_state_batch = torch.from_numpy(transitions[3]).to(device)
    done_batch_mask = transitions[4] == 0.0

    # 2. compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

    # 3 & 4. compute \max_a Q(s_{t+1}, a) for all next states and mask were episodes have terminated
    next_state_values = torch.zeros(batch_size, device=device)

    if double_dqn:
        with torch.no_grad():
            next_state_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_state_values[done_batch_mask] = target_net(next_state_batch[done_batch_mask]).gather(1, next_state_actions[done_batch_mask]).squeeze(1)
    else:
        with torch.no_grad():
            next_state_values[done_batch_mask] = target_net(next_state_batch[done_batch_mask]).max(1)[0].detach()

    # 5. compute the target Q values
    expected_state_action_values = reward_batch + gamma * next_state_values

    # 6. compute loss (Huber)
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 7. calculate the gradients
    optimizer.zero_grad()
    loss.backward()

    # 8. clip the gradients
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)

    # 9. optimize
    optimizer.step()

    return loss.item()

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network

    target_net.load_state_dict(policy_net.state_dict())


    
