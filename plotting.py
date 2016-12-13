import numpy as np
from pylab import *
import matplotlib.pyplot  as pyplot
import matplotlib.patches as mpatches

font = {'size':16}
plt.rc('font', **font)

def parse(fileNames, titleName):
    patches = []
    for fileName, marker, color, label in fileNames:
        with open('edward_logs/' + fileName + '.log', 'r') as f:
            read_data = f.read()
        val_losses = []
        for line in read_data.split("\n"):
            if "Val Loss" in line:
                val_losses.append(line.split()[-1])
        Y = np.asarray(val_losses)[:40]
        X = np.arange(len(Y))
        plt.plot(X, Y, marker)
        patches.append(mpatches.Patch(color=color, label=label))
    pyplot.yscale('log')
    plt.xlabel('Epoch Number')
    plt.ylabel('Validation Error')
    plt.title(titleName)
    plt.legend(handles=patches)
    plt.show()

# Plots the data given in X and Y
# X and Y must be vectors of the same size
def plot(X, Y, xlabel, ylabel, title):
    plt.plot(X,Y,'ro')
    pyplot.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plotNoise():
    parse([
    ("smaller", "bo", "blue", "Without Noise"),
    ("smallerwithnoise", "ro", "red", "With Noise")
    ],
    "Val Error for With and Without Noise")   

def plotDense():
    parse([
    ("bigger", "bo", "blue", "Three Dense"),
    ("standard", "ro", "red", "Two Dense"),
    ("smaller", "go", "green", "One Dense")
    ],
    "Val Error for Varying Numbers of Dense Layers")

def plotLSTM():
    parse([
    ("evenmorelstm", "bo", "blue", "Three LSTM"),
    ("morelstm", "ro", "red", "Two LSTM"),
    ("smaller", "go", "green", "One LSTM")
    ],
    "Val Error for Varying Numbers of LSTM Layers")

def plotLayerSize():
    parse([
    ("bigsize", "bo", "blue", "Large Sized"),
    ("smaller", "ro", "red", "Medium Sized"),
    ("smallsize", "go", "green", "Small Sized")
    ],
    "Val Error for Varying Sizes of Layers")

plotLayerSize()