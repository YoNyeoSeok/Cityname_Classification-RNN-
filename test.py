from data import * 
from model import *
from plot import *
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='logs/')
args = parser.parse_args()

all_letters, all_categories, category_lines, val_category_lines = getData()
data2tensor = Data2Tensor(all_letters, all_categories)

all_losses = np.load(args.log_dir+'all_losses.npy')
all_val_losses = np.load(args.log_dir+'all_val_losses.npy')
model = torch.load(args.log_dir+'model')

fig = plot_losses(all_losses, all_val_losses)
plt.savefig(args.log_dir+'losses')
confusion = np.zeros((data2tensor.n_categories, data2tensor.n_categories))
for category, lines in val_category_lines.items():
    val_category_idx = all_categories.index(category)
    for line in lines:
        val_category_tensor, val_line_tensor = data2tensor.dataToTensor(val_category_idx, line)
    #    val_category, val_line, val_category_tensor, val_line_tensor = randomValExample()
        val_output, _ = model.test(val_line_tensor, val_category_tensor)
    
        guess_idx = np.argmax(val_output)
        confusion[val_category_idx][guess_idx] += 1
    
accuracy = np.diag(confusion).sum() / confusion.sum()
for i in range(data2tensor.n_categories):
    confusion[i] /= confusion[i].sum()
fig = plot_confusion_matrix(confusion, data2tensor.all_categories, title='confusion_matrix' + 
        '\naccuracy: %.3f'%(accuracy))
plt.savefig(args.log_dir + 'confusion_matrix')

