"""
PlotCode:
    Defines the Plot class to save plot images of the metrics from the trianed model
"""
import matplotlib.pyplot as plt

class Plot:

    """
    Class to plot graphs
    """

    def __init__(self, y, x, param, color):
        """
        Input:
            y: values set for the chosen metric
            x: length of the set y
            param: chosen comaparison metric for the model
            color: line color for set y

        Action:
            Initialized the input arguments to class variables
        """
        self.y = y
        self.x = x
        self.param = param
        self.color = color

    def __call__(self):
        """
        Plots the graph and saves it to the 'Images/' directory
        """
        y, x, param, color = self.y, self.x, self.param, self.color
        plt.figure(figsize=(50, 30), dpi = 120)
        plt.plot(x, y, color, linewidth = 3)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.legend([param], loc = 'upper right', fontsize = 24)
        plt.title(r"Comparison of "+param, fontsize = 36)
        plt.xlabel(r"Number of Epochs", fontsize = 24)
        plt.ylabel("Parameters", fontsize = 24)
        plt.savefig("Images/"+param+".png")
