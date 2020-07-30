import pandas as pd
import numpy as np
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define constants
TEST_REGISTERS = 90
WINDOWS_PAST_STATES = 30

DATASETS = [
    'data/bitcoin_price.csv',
    'data/ethereum_price.csv',
    'data/litecoin_price.csv'
    ]

MONTHS = [ 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def load_data_unormalized(test_registers=TEST_REGISTERS):

    dfs = []

    for dataset in DATASETS:
        dfs.append(pd.read_csv(dataset))

    for idx, df in enumerate(dfs):
        df = df.add_suffix('_{}'.format(idx))
        df['Year'] = df['Date_{}'.format(idx)].apply(lambda x: int(x.split(',')[1]))
        df['Month'] = df['Date_{}'.format(idx)].apply(lambda x: x.split(',')[0].split()[0])
        df['Month'] = df['Month'].apply(lambda x: MONTHS.index(x) + 1)
        df['Day'] = df['Date_{}'.format(idx)].apply(lambda x: int(x.split(',')[0].split()[1]))
        dfs[idx] = df.sort_values(['Year', 'Month', 'Day'], ascending=True)
    
    df = pd.concat([df.set_index(['Year', 'Month', 'Day']) for df in dfs], axis=1).reset_index()
    df = df[~df.isnull().any(axis=1)]
    data_open = []
    data_close = []

    for idx in range(len(DATASETS)):
        values = df[['Open_{}'.format(idx), 'Close_{}'.format(idx)]].values
        data_close.append(values[:, 1])
        data_open.append(values[:, 0])

    data_close = np.stack(data_close, axis=-1)
    data_open = np.stack(data_open, axis=-1)

    train_close = data_close[:-TEST_REGISTERS]
    test_close = data_close[-TEST_REGISTERS - WINDOWS_PAST_STATES + 1:]


    train_open = data_open[:-TEST_REGISTERS]
    test_open = data_open[-TEST_REGISTERS - WINDOWS_PAST_STATES + 1:]

    return train_open, test_open, train_close, test_close

def load_data(test_registers=TEST_REGISTERS):

    dfs = []

    for dataset in DATASETS:
        dfs.append(pd.read_csv(dataset))

    for idx, df in enumerate(dfs):
        df = df.add_suffix('_{}'.format(idx))
        df['Year'] = df['Date_{}'.format(idx)].apply(lambda x: int(x.split(',')[1]))
        df['Month'] = df['Date_{}'.format(idx)].apply(lambda x: x.split(',')[0].split()[0])
        df['Month'] = df['Month'].apply(lambda x: MONTHS.index(x) + 1)
        df['Day'] = df['Date_{}'.format(idx)].apply(lambda x: int(x.split(',')[0].split()[1]))
        dfs[idx] = df.sort_values(['Year', 'Month', 'Day'], ascending=True)
    
    df = pd.concat([df.set_index(['Year', 'Month', 'Day']) for df in dfs], axis=1).reset_index()
    df = df[~df.isnull().any(axis=1)]
    data = []

    for idx in range(len(DATASETS)):
        values = df[['Open_{}'.format(idx), 'Close_{}'.format(idx)]].values
        data.append(values[:, 1] / values[:, 0])

    data = np.stack(data, axis=-1)

    train = data[:-TEST_REGISTERS]
    test = data[-TEST_REGISTERS - WINDOWS_PAST_STATES + 1:]

    return train, test


################################################################################
# Environment
################################################################################ 


State = namedtuple('State', ('past_values', 'invested_amount', 'current_amount'))

class Environment():
    def __init__(self, data, window_past_states=WINDOWS_PAST_STATES, initial_amount=1000, max_loss=0.3, action_step=10):
        self.data = data
        self.num_assets = self.data.shape[1]
        self.len_episode = self.data.shape[0]

        self.actions = []
        
        for i in range(0, 100 + action_step, action_step):
            for j in range(0, 100 + action_step, action_step):
                for k in range(0, 100 + action_step, action_step):
                    for l in range(0, 100 + action_step, action_step):
                        if (i + j + k + l) == 100:
                            self.actions.append([i / 100, j / 100, k / 100, l / 100])

        self.actions = np.array(self.actions, dtype=np.float)

        self.n_actions = self.actions.shape[0]

        self.initial_amount = initial_amount
        self.current_amount = None
        self.curr_step = None
        self.min_amount = initial_amount * (1 - max_loss)
        self.window_past_states = window_past_states
        self.state = None
        self.done = False

        #self.fees = np.array([0.003] * self.num_assets + [0])
        self.fees = np.array([0.01] * self.num_assets + [0])

    def get_past_values(self):
        return np.array(np.expand_dims(self.data[self.curr_step:self.curr_step + self.window_past_states], axis=-1), dtype=np.float)

    def restart(self):
        self.curr_step = 0
        self.done = False
        self.current_amount = self.initial_amount
        
        past_values = self.get_past_values()
        invested = np.array([0] * self.num_assets + [self.initial_amount], dtype=np.float)
        self.state = State(past_values, invested, self.current_amount)

        return self.state
    
    # 1% Percent of investment
    def get_fee(self, action):
        return (action * self.fees).sum(axis=-1) * self.current_amount

    # Loss for action taken 
    def get_loss(self, action): 
        
        investment = action * self.current_amount
        ratios = np.concatenate([self.data[self.curr_step + self.window_past_states + 1], [1]])
        loss = self.current_amount - (investment * ratios).sum(axis=-1)
        
        return loss

    def get_optimals(self):
        try:
            fees = self.get_fee(self.actions)
            losses = self.get_loss(self.actions)
            costs = fees + losses
            optimal_cost = np.min(1 - (self.current_amount - costs) / self.current_amount)
            optimal_action_idx = np.argmin(1 - (self.current_amount - costs) / self.current_amount)

            return optimal_action_idx, optimal_cost
        except:
            return None, None


    # Calculates immediate loss, update state and decide if done
    def step(self, action_idx):

        # Decide if done
        self.done = (self.state.current_amount <= self.min_amount) or (self.curr_step + self.window_past_states + 1 >= self.len_episode)

        if self.done:
            return None, 0, self.done, self.curr_step

        action = self.actions[action_idx]
        # Immediate cost
        cost = self.get_fee(action) + self.get_loss(action)  
        cost = 1 - (self.current_amount - cost) / self.current_amount
        # Update state
        self.update_state(action, cost)
        
        
        return self.state, cost, self.done, self.curr_step
        
    def update_state(self, action, cost):
        # Update step
        self.curr_step += 1

        past_values = self.get_past_values()
        # Calculate invested
        invested = action * self.current_amount
        # Disccount fees
        invested = invested - (action * self.current_amount * self.fees)
        self.current_amount = self.current_amount * (1 - cost)
        self.state = State(past_values, invested, self.current_amount)
       
        return self.state
        
################################################################################
# Relpay memory
################################################################################ 
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'cost'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Saves a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


################################################################################
# Q-values approximantor
################################################################################ 

class DQN(nn.Module):        
    def __init__(self, num_actions, dropout=0.2):        
        super(DQN, self).__init__()

        self.rnn_0 = nn.GRU(1, 30, batch_first=True, num_layers=1) 
        self.linear_0 = nn.Linear(30, 128)

        self.rnn_1 = nn.GRU(1, 30, batch_first=True, num_layers=1) 
        self.linear_1 = nn.Linear(30, 128)

        self.rnn_2 = nn.GRU(1, 30, batch_first=True, num_layers=1) 
        self.linear_2 = nn.Linear(30, 128)

        self.linear_3 = nn.Linear(4, 128)
        self.activation_3 = nn.Tanh()

        self.linear_4 = nn.Linear(512, 1024)
        self.activation_4 = nn.Sigmoid()
        self.dropout_4 = nn.Dropout(dropout)

        self.linear_5 = nn.Linear(1024, num_actions)
        self.activation_5 = nn.Tanh()
        
    def forward(self, preds, invested):      

        out_0, out_1, out_2, out = preds[:, :, 0], preds[:, :,  1], preds[:, :,  2], invested

        out_0, _= self.rnn_0(out_0)
        out_0 = self.linear_0(out_0)

        out_1, _ = self.rnn_1(out_1)
        out_1 = self.linear_1(out_1)

        out_2, _ = self.rnn_2(out_2)
        out_2 = self.linear_2(out_2)

        out = self.linear_3(out)
        out = self.activation_3(out)

        out = torch.cat([
            out_0[:, -1].squeeze(dim=1), 
            out_1[:, -1].squeeze(dim=1), 
            out_2[:, -1].squeeze(dim=1), 
            out
            ], axis=-1)

        out = self.linear_4(out)
        out = self.activation_4(out)
        out = self.dropout_4(out)

        out = self.linear_5(out)
        out = self.activation_5(out)

        return(out)

################################################################################
# Epsilon greedy policy
################################################################################ 
def get_select_action_fn(policy_net, n_actions, device, eps_start=0.9, eps_end=0.05, eps_decay=200):
    def select_action(state, current_step):
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * current_step / eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(
                    torch.from_numpy(np.array([state.past_values])).float().to(device), 
                    torch.from_numpy(np.array([state.invested_amount])).float().to(device)
                ).min(dim=1)[1].detach().cpu().numpy().item()
        else:
            return random.randrange(n_actions)

    return select_action




################################################################################
# Optimization step
################################################################################ 


('past_values', 'invested_amount', 'current_amount')

def get_optimization_fn(policy_net, target_net, memory, optimizer, device, batch_size=128, gamma=0.999):
    def optimize_model():
        if len(memory) < batch_size:
            return 0

        transitions = memory.sample(batch_size)

        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
                            tuple(
                                map(
                                    lambda s: s is not None,
                                    batch.next_state
                                )
                            ), device=device, dtype=torch.bool
                        )

        non_final_next_states = (
                torch.stack([ torch.from_numpy(s.past_values) for s in batch.next_state if s is not None ]).float().to(device), 
                torch.stack([ torch.from_numpy(s.invested_amount) for s in batch.next_state if s is not None ]).float().to(device)
            )
        
        state_batch = (
                torch.stack([ torch.from_numpy(s.past_values) for s in batch.state if s is not None ]).float().to(device), 
                torch.stack([ torch.from_numpy(s.invested_amount) for s in batch.state if s is not None ]).float().to(device)
            )
        
        

        #state_batch = (torch.cat(batch.state))
        action_batch = torch.cat([torch.Tensor([[a]]) for a in batch.action]).long().to(device)
        cost_batch = torch.cat([torch.Tensor([c]) for c in batch.cost]).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = policy_net(state_batch[0], state_batch[1]).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states[0], non_final_next_states[1]).min(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + cost_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        #for param in policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss.item()

    return optimize_model


################################################################################
# Saving
################################################################################ 

MODEL_STATE_DICT = 'model_state_dict'
OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
EPOCH_STATE = 'epoch'
LOSS_HISTORY_STATE = 'loss_hist'
COST_HISTORY_STATE = 'cost_hist'
DURATION_HISTORY_STATE = 'duration_hist'

def save_model_state(filename, model, optimizer, epoch, loss_hist, cost_hist, durations_hist):
    torch.save({
        MODEL_STATE_DICT: model.state_dict(),
        OPTIMIZER_STATE_DICT: optimizer.state_dict(),
        EPOCH_STATE: epoch,
        LOSS_HISTORY_STATE: loss_hist,
        COST_HISTORY_STATE: cost_hist,
        DURATION_HISTORY_STATE: durations_hist
    }, filename)

def load_model_state(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT])
    last_epoch = checkpoint[EPOCH_STATE]
    loss_history = checkpoint[LOSS_HISTORY_STATE]
    cost_history = checkpoint[COST_HISTORY_STATE]
    duration_history = checkpoint[DURATION_HISTORY_STATE]

    return model, optimizer, last_epoch, loss_history, cost_history, duration_history