import matplotlib.pyplot as plt
import numpy as np

def stdplot(x, y, Nsig=2, ax=None, color='b', axis=1, **kargs):
    """
    Plot the mean / standard-deviation for a signal
    
    x: x values
    y: y values. One of the axis is used for the statistics
    axis: which axis to calculate mean/std
    ax: maplotlib axis to plot in
    color: color of the line
    **kargs: arguments passed to plot
    
    """
	if ax is None:
		ax = plt.gca()
	dy = np.std(y, axis=axis)
	mean = np.mean(y, axis=axis)
	ax.plot(x, mean, color=color, **kargs)
	for i in range(1, Nsig+1):
		ax.fill_between(x, mean-i*dy, mean+i*dy, color=color, alpha=.2)

def sublegend(*ax, labels=None, color='white', margin=9, titles=None, fontsize=14):
    """
    ax: list of axes
    labels: If None, the will be a, b, c, d, ...
    color: background color of the rectangle
    margin: margin in pixel from the axis border
    titles: set to False to remove all titles. (Useful to keep the set_title info in the code to remember what is plotted)
    fontsize: The font size of the labels
    """
    props = dict(boxstyle='round', facecolor=color, alpha=1)  
    if not hasattr(margin, "__getitem__") and not hasattr(margin, "__iter__"):
        margin = (margin, margin)

    if labels is None:
        labels = [chr(ord('a')+i) for i,_ in enumerate(np.ravel(ax))]
    for i,a in enumerate(np.ravel(ax)):
        if titles is False:
            a.set_title("")
        a.annotate(labels[i],(0, 1),xytext=(margin[0],-margin[1]),fontsize=fontsize,verticalalignment='top', bbox=props, xycoords='axes fraction',textcoords='offset pixels');
        
def plotFWHM(ax, x0, G, h, fmt=".2f", unit='px', col='k', linestyle='-', offset=(0,0), va="bottom"):
    lab = "{:"+fmt+"}"+unit
    ax.annotate("",(x0-G/2,h),(x0+G/2,h), arrowprops=dict(arrowstyle='<|-|>',color=col, linestyle=linestyle))
    ax.annotate(lab.format(G),(x0,h),offset, color=col, textcoords='offset pixels', ha="center", va=va)
    
def plotMask(ax, mask, color, **kargs):
    """
    Plot a masked region with a specific color
    ax: axis
    mask: An array with value True where the mask should be plotted
    color: color of the mask
    **kargs: dictionnary of arguments which can be passed to imshow. Useful is mainly: alpha
    """
    import copy
    m = np.ma.masked_array(mask, ~mask)
    palette = copy.copy(plt.cm.gray)
    palette.set_over(color, 1.0)
    ax.imshow(m, cmap=palette, vmin=0, vmax=0.5, **kargs)
    