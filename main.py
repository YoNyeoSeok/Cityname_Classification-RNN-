"""
python main.py --model RNN --log_dir logs/rnn/
python main.py --model LSTM --log_dir logs/lstm/
python main.py --model tanh --log_dir logs/tanh/
"""

from data import * 
from model import *
from plot import *
import numpy as np
import time
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='RNN')
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--n_iters', type=int, default=100000)
parser.add_argument('--print_every', type=int, default=5000)
parser.add_argument('--plot_every', type=int, default=1000)
args = parser.parse_args()

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)

# getData
all_letters, all_categories, category_lines, val_category_lines = getData()
trainDataBatch = GetDataBatch(all_categories, category_lines)
validDataBatch = GetDataBatch(all_categories, val_category_lines)
data2tensor = Data2Tensor(all_letters, all_categories)

# buildModel
hidden_size = 128
#num_layers = 1
if 'RNN' == args.model:
    model = RNN(input_size=data2tensor.n_letters, 
            hidden_size=hidden_size, 
            #num_layers=num_layers,
            output_size=data2tensor.n_categories)
elif 'LSTM' == args.model:
    model = LSTM(input_size=data2tensor.n_letters, 
            hidden_size=hidden_size, 
            #num_layers=num_layers,
            output_size=data2tensor.n_categories)
elif 'GRU' == args.model:
    model = GRU(input_size=data2tensor.n_letters, 
            hidden_size=hidden_size, 
            #num_layers=num_layers,
            output_size=data2tensor.n_categories)
 
    
learning_rate = 1e-3 # If you set this too high, it might explode. If too low, it might not learn
model.set_optimizer(optimizer='Adam', learning_rate=learning_rate)

# Keep track of losses for plotting
current_loss = 0
val_current_loss = 0
all_losses = []
all_val_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

batch_size = 100
start = time.time()

for iter in range(1, args.n_iters + 1):
    category, category_idx, line = randomDataExample(all_categories, category_lines)
    category_tensor, line_tensor = data2tensor.dataToTensor(category_idx, line)
    output, loss = model.train(line_tensor, category_tensor)
    current_loss += loss
    
    val_category, val_category_idx, val_line = randomDataExample(all_categories, val_category_lines)
    val_category_tensor, val_line_tensor = data2tensor.dataToTensor(val_category_idx, val_line)
    val_output, val_loss = model.test(val_line_tensor, val_category_tensor)
    val_current_loss += val_loss

    # Print iter number, loss, name and guess
    if iter % args.print_every == 0:
        guess_idx = np.argmax(val_output)
        guess = data2tensor.categoryFromIdx(guess_idx)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / args.n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % args.plot_every == 0:
        all_losses.append(current_loss / args.plot_every)
        all_val_losses.append(val_current_loss / args.plot_every)
        current_loss = 0
        val_current_loss = 0

np.save(args.log_dir+'all_losses', all_losses)
np.save(args.log_dir+'all_val_losses', all_val_losses)
model.save(args.log_dir)

fig = plot_losses(all_losses, all_val_losses)
plt.savefig(args.log_dir+'losses')
confusion = np.zeros((data2tensor.n_categories, data2tensor.n_categories))
for category, lines in val_category_lines.items():
    val_category_idx = all_categories.index(category)
    for line in lines:
        val_category_tensor, val_line_tensor = data2tensor.dataToTensor(val_category_idx, line)
        val_output, _ = model.test(val_line_tensor, val_category_tensor)
    
        guess_idx = np.argmax(val_output)
        confusion[val_category_idx][guess_idx] += 1
    
accuracy = np.diag(confusion).sum() / confusion.sum()
for i in range(data2tensor.n_categories):
    confusion[i] /= confusion[i].sum()
fig = plot_confusion_matrix(confusion, data2tensor.all_categories, title='confusion_matrix' + 
        '\naccuracy: %.3f'%(accuracy))
plt.savefig(args.log_dir + 'confusion_matrix')

