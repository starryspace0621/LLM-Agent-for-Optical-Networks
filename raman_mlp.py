import json
from tqdm import tqdm
import os
import random
import ast
import Raman_simulator_for_LLM.scripts.propagation_Raman as raman
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = 'cpu'

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hiddens_size, activation_fn):
        super(MLP, self).__init__()
        layers = []
        size = input_size
        for hidden_size in hiddens_size:
            layers.append(nn.Linear(size, hidden_size))
            layers.append(self.get_activation_fn(activation_fn))
            size = hidden_size
        layers.append(nn.Linear(size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    def get_activation_fn(self, activation_fn):
        if activation_fn == 'relu':
            return nn.ReLU()
        elif activation_fn == 'tanh':
            return nn.Tanh()
        elif activation_fn == 'sigmoid':
            return nn.Sigmoid()
        elif activation_fn == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        
    
def ave_5(x):
    ave = []
    if not isinstance(x, list):
        x = x.tolist()
    for i in range(len(x) // 5):
        ave.append(sum(x[i*5:i*5+5]) / 5)
    return torch.tensor(ave)

def mlp_train(input_file, output_file, criter, optimal, hidden_size, activation_fn):
    with open(input_file, 'r') as f:
        inputs = json.load(f)
    inputs = torch.tensor(inputs).to(DEVICE)
    with open(output_file, 'r') as f:
        outputs = json.load(f)
    outputs = torch.tensor(outputs).to(DEVICE)

    dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    input_size = inputs.shape[1]
    output_size = outputs.shape[1]
    # hidden_size = [64, 128]
    model = MLP(input_size, output_size, hidden_size, activation_fn).to(DEVICE)
    criterion = getattr(nn, criter)()
    optimizer = getattr(optim, optimal)(model.parameters(), lr=0.01)

    num_epochs = 200
    for epoch in tqdm(range(num_epochs), leave=False):
        for input, label in dataloader:
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    SAVE_FILE = f'{hidden_size},{activation_fn},{criter},{optimal},{loss.item():.4f}.pth'
    SAVE_FILE = SAVE_FILE.replace(' ', '')
    torch.save(model.state_dict(), SAVE_FILE)
    return SAVE_FILE


def mlp_test(pth, input_size=6, output_size=10):
    if not os.path.exists(pth):  
        raise FileNotFoundError(f"The file '{pth}' does not exist.")

    input_str = pth.strip().strip("'\"")
    input_str = input_str.replace(' ', '')
    try:
        hiddens_sizes_str, activation, criterion, optimizer, loss = input_str.rsplit(',', 4)
        hiddens_sizes = ast.literal_eval(hiddens_sizes_str)
    except (ValueError, SyntaxError):
        try:
            parsed_input = json.loads(f"[{input_str}]")
            hiddens_sizes, activation, criterion, optimizer = parsed_input
            hiddens_sizes = ast.literal_eval(hiddens_sizes)
        except json.JSONDecodeError:
            raise ValueError("Unable to parse input string.")

    model = MLP(input_size, output_size, hiddens_sizes, activation).to(DEVICE)
    model.load_state_dict(torch.load(pth))
    criterion = getattr(nn, criterion)()

    TEST_SIZE = 12
    error = 0
    for i in tqdm(range(TEST_SIZE)):
        x = torch.tensor([random.uniform(0, 0.25) for _ in range(6)]).to(DEVICE)
        y_pre = model(x)
        y_true = ave_5(raman.raman_transmit(x.tolist()))
        error += criterion(y_pre, y_true)
    
    return (error / TEST_SIZE).tolist()

def mlp_pred(input_v, pth, input_size=6, output_size=10):
    if not isinstance(input_v, torch.Tensor):
        input_v = torch.tensor(input_v).to(DEVICE)
    
    # if not os.path.exists(pth):
    #     raise FileNotFoundError(f"The file '{pth}' does not exist.")
    input_str = pth.strip().strip("'\"")
    input_str = input_str.replace(' ', '')
    try:
        hiddens_sizes_str, activation, criterion, optimizer, loss = input_str.rsplit(',', 4)
        hiddens_sizes = ast.literal_eval(hiddens_sizes_str)
    except (ValueError, SyntaxError):
        try:
            parsed_input = json.loads(f"[{input_str}]")
            hiddens_sizes, activation, criterion, optimizer = parsed_input
            hiddens_sizes = ast.literal_eval(hiddens_sizes)
        except json.JSONDecodeError:
            raise ValueError("Unable to parse input string.")

    model = MLP(input_size, output_size, hiddens_sizes, activation).to(DEVICE)
    model.load_state_dict(torch.load(pth))
    outs = model(input_v).tolist()
    return [round(out, 4) for out in outs]