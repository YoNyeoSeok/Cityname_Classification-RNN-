import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

class Data2Tensor():
    def __init__(self, all_letters, all_categories):
        self.all_letters = all_letters
        self.all_categories = all_categories

        self.n_letters = len(all_letters)
        self.n_categories = len(all_categories)

    # Find letter index from self.all_letters, e.g. "a" = 0
    def letterToIndex(self, letter):
        return self.all_letters.find(letter)
    
    # Just for demonstration, turn a letter into a <1 x self.n_letters> Tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor
    
    # Turn a line into a <line_length x 1 x self.n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

    def categoryFromIdx(self, category_idx):
        return self.all_categories[category_idx]
    
    def dataToTensor(self, category_idx, line):
        category_tensor = torch.tensor([category_idx], dtype=torch.long)
        line_tensor = self.lineToTensor(line)
        return category_tensor, line_tensor

    def batchDataToPackSequence(self, category_idxs, lines):
        #        batch_category_tensor = PackedSequence(torch.tensor(
#            [torch.tensor(category_idx, dtype=torch.long) for category_idx in category_idxs]))
        batch_lines_tensor = PackedSequence(torch.cat([self.lineToTensor(line) for line in lines]))
        batch_category_tensor = PackedSequence(torch.tensor(category_idxs))
        return batch_category_tensor, batch_line_tensor

# RNN = nn.RNN
class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.criterion = None
        self.optimizer = None
    
    def set_optimizer(self, optimizer, learning_rate):
        if 'Adam' == optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def save(self, PATH):
        torch.save(self, PATH+'model')

    def load(self, PATH):
        self = torch.load(PATH+'model')


class RNN(ModelBase):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.criterion = nn.NLLLoss()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.tanh(hidden)
        output = self.i2o(combined)
        output = self.logsoftmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def train(self, line_tensor, category_tensor):
        hidden = self.initHidden()
    
        self.zero_grad()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
    
        loss = self.criterion(output, category_tensor)
        loss.backward()
    
        # Add parameters' gradients to their values, multiplied by learning rate
        self.optimizer.step()
    #     for p in rnn.parameters():
    #         print(p.grad.data)
    #         p.data.add_(-learning_rate, p.grad.data)
    
        return output.data.numpy(), loss.item()
    
    def test(self, line_tensor, category_tensor):
        hidden = self.initHidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
    
        loss = self.criterion(output, category_tensor)
        
        return output.data.numpy(), loss.item()

class LSTM(ModelBase):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.gate_size = 4*hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, self.gate_size)
        self.sigmoid = nn.Sigmoid()

        self.tanh = nn.Tanh()

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.criterion = nn.NLLLoss()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        h_g3 = self.i2h(combined)
        h, ig, fg, og = torch.split(h_g3, [self.hidden_size, self.hidden_size,
                                          self.hidden_size, self.hidden_size], dim=1)
        ig = self.sigmoid(ig)
        fg = self.sigmoid(fg)
        og = self.sigmoid(og)
        h = self.tanh(h)
        hidden = fg * hidden + ig * h
        hidden = og * self.tanh(hidden)

        output = self.i2o(combined)
        output = self.logsoftmax(output)
#         print(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def train(self, line_tensor, category_tensor):
        hidden = self.initHidden()
    
        self.zero_grad()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
    
        loss = self.criterion(output, category_tensor)
        loss.backward()
    
        # Add parameters' gradients to their values, multiplied by learning rate
        self.optimizer.step()
    #     for p in rnn.parameters():
    #         print(p.grad.data)
    #         p.data.add_(-learning_rate, p.grad.data)
    
        return output.data.numpy(), loss.item()
    
    def test(self, line_tensor, category_tensor):
        hidden = self.initHidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
    
        loss = self.criterion(output, category_tensor)
        
        return output.data.numpy(), loss.item()

class GRU(ModelBase):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.gate_size = 2*hidden_size

        self.i2g = nn.Linear(input_size + hidden_size, self.gate_size)
        self.i2c = nn.Linear(input_size, self.hidden_size)
        self.h2c = nn.Linear(hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.tanh = nn.Tanh()

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.criterion = nn.NLLLoss()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        g2 = self.i2g(combined)
        rg, zg = torch.split(g2, [self.hidden_size, self.hidden_size], dim=1)
        rg = self.sigmoid(rg)
        zg = self.sigmoid(zg)
        ic = self.i2c(input)
        hc = self.h2c(hidden)
        h = self.tanh(ic+rg*hc)
        hidden = (1-zg) * h + zg * hidden

        output = self.i2o(combined)
        output = self.logsoftmax(output)
#         print(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def train(self, line_tensor, category_tensor):
        hidden = self.initHidden()
    
        self.zero_grad()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
    
        loss = self.criterion(output, category_tensor)
        loss.backward()
    
        # Add parameters' gradients to their values, multiplied by learning rate
        self.optimizer.step()
    #     for p in rnn.parameters():
    #         print(p.grad.data)
    #         p.data.add_(-learning_rate, p.grad.data)
    
        return output.data.numpy(), loss.item()
    
    def test(self, line_tensor, category_tensor):
        hidden = self.initHidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
    
        loss = self.criterion(output, category_tensor)
        
        return output.data.numpy(), loss.item()



        
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(RNN, self).__init__()
# 
#         self.hidden_sizes = hidden_sizes
#         self.i2hs = [nn.Linear(input_size + hidden_sizes[0], hidden_size[0])]
#         for _hidden_size, hidden_size_ in zip(hidden_sizes[1:], hidden_sizes[2:]+[output_size]):
#             self.i2hs += [nn.Linear(_hidden_size, hidden_size_)]
#         self.i2o = nn.Linear(hidden_sizes[-1], output_size)
#         self.tanh = nn.Tanh()
#         self.softmax = nn.LogSoftmax(dim=1)
# 
#     def forward(self, input, hiddens):
#         combined = torch.cat((input, hiddens[0]), 1)
#         for i2h in self.i2hs:
#             hidden = i2h(combined)
#             hidden = self.tanh(hidden)
#             combined = hidden
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden
# 
#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size)
# 
# def RNN(nn.RNN):
#     def __init__(self, *args, **kwargs):
#         if 'output_size' in kwargs:
#             self.output_size = kwargs.pop('output_size')
#         elif len(args) > 2:
#             self.output_size = args.pop(2)
# 
#         if 'hidden_sizes' in kwargs:
#             self.hidden_sizes = kwargs.pop('hidden_sizes')
#         elif 'hidden_size' in kwargs:
#             self.hidden_sizes = [kwargs.pop('hidden_size')]
#         elif len(args) > 1:
#             if type(args[2]) == list:
#                 self.hidden_sizes = [args.pop(1)]
#             else:
#                 self.hidden_sizes = args.pop(1)
# 
#         rnns = []
#         for hidden_size in self.hidden_sizes:
#             kwargs.update({'hidden_size':hidden_size})
#             rnns += [super(RNN, self).__init__(*args, **kwargs)]
# 
#     def forward(self, input, hx=None):
#         output = input
#         hidden = hx
#         for rnn in rnns:
#             output, hidden = super(RNN, self).forward(ouput, hidden)
# 
#         return output, hidden
# 
