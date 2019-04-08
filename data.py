import glob
import os
import codecs
import numpy as np
import random

def findFiles(path): return glob.glob(path)

# Read a file and split into lines
def readLines(filename):
    lines = codecs.open(filename, "r", encoding='utf-8', 
                        errors='ignore').read().strip().split('\n')
    return [line for line in lines]
       
def getData(train_path='train/*.txt', val_path='val/*.txt'): 
    # Build the category_lines dictionary, a list of names per language
    all_letters = ""
    all_categories = []
    
    category_lines = {}
    val_category_lines = {}
    
    for filename in findFiles(train_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        all_letters += "".join(lines)
        category_lines[category] = lines
    
    for filename in findFiles(val_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        lines = readLines(filename)
        all_letters += "".join(lines)
        val_category_lines[category] = lines
    
    n_categories = len(all_categories)
    all_letters = "".join(np.unique([letter for letter in all_letters]))
    n_letters = len(all_letters)

    return all_letters, all_categories, category_lines, val_category_lines

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomDataExample(all_categories, category_lines):
    category = randomChoice(all_categories)
    category_idx = all_categories.index(category)
    line = randomChoice(category_lines[category])
    return category, category_idx, line

class GetDataBatch():
    def __init__(self, all_categories, category_lines):
        self.all_categories = all_categories
        self.category_lines = category_lines
        #self.data = [(line, category) for line in lines for category, lines in category_lines.items()]
        self.data = np.array([(category, all_categories.index(category), line) 
                for category, lines in category_lines.items()
                for line in lines], dtype=("U2, i, U30"))
       
        self.shuffle()
        self.loc = 0
    def shuffle(self):
        random.shuffle(self.data)

    def getBatch(self, batch_size=100):
        if self.loc + batch_size > len(self.data):
            self.loc = 0
        self.loc += batch_size
        return self.data[self.loc-batch_size:self.loc]

    def dataSplit(self, data):
        categories, category_idxs, lines = np.array(list(zip(*data)))
        return categories, category_idxs.astype(np.int), lines
