import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('--log_dir', type=str, default='logs/')
#args = parser.parse_args()
#
def load_losses(all_losses_path, all_val_losses_path):
    all_losses = np.load(all_losses_path)
    all_val_losses = np.load(all_val_losses_path)
    return all_losses, all_val_losses

def plot_loss(ax, losses, label='loss'):
    ax.plot(losses, label=label)

def plot_losses(all_losses, all_val_losses, title='losses'):
    #    all_losses, all_val_losses = load_losses(args.log_dir+'all_losses', args.log_dir+'all_val_losses') 
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111)
    plot_loss(ax, all_losses, 'training_loss')
    plot_loss(ax, all_val_losses, 'validation_loss')
    ax.legend()
    return fig
    plt.show()
    fig.save(title+'.jpg')

def confusion_matrix(fig, classes):
	confusion = np.zeros(classes, classes)
	
def plot_confusion_matrix(confusion, class_names, title='confusion_matrix'):
    assert len(confusion.shape) == 2
    assert confusion.shape[0] == confusion.shape[1]
    # Set up plot
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)
    
    # Set up axes
    ax.set_xticklabels([''] + class_names, rotation=90)
    ax.set_yticklabels([''] + class_names)
    
    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # sphinx_gallery_thumbnail_number = 2
    return fig

"""
#plt.show()
plt.figure()
plt.plot(all_losses)
plt.plot(all_val_losses)

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomValExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

"""
