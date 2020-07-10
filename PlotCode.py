import matplotlib.pyplot as plt

def plot(y, x, param, color):
    plt.figure(figsize=(50, 30), dpi = 120)
    plt.plot(x, y, color, linewidth = 3)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend([param], loc = 'upper right', fontsize = 24)
    plt.title(r"Comparison of "+param, fontsize = 36)
    plt.xlabel(r"Number of Epochs", fontsize = 24)
    plt.ylabel("Parameters", fontsize = 24)
    plt.savefig("Images/"+param+".png")