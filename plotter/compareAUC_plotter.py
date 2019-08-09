import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

workPath = "/work/kimmokal/qgDNN"
outPath = workPath+"/plotter/plots/"

# This is a very clumsy script, as it requires the values to be put in manually!
QGl_AUC_1 = [0.753, 0.793, 0.801, 0.805]
QGl_AUC_2 = [0.728, 0.772, 0.793]
fNN_AUC_1 = [0.751, 0.788, 0.793, 0.802]
fNN_AUC_2 = [0.732, 0.777, 0.792]
DJ_AUC_1 = [0.774, 0.819, 0.840, 0.852]
DJ_AUC_2 = [0.757, 0.806, 0.831]
Img_AUC_1 = [0.790, 0.828, 0.842, 0.852]
Img_AUC_2 = [0.761, 0.804, 0.831]

pt1 = [65, 200, 650, 1000]
pt2 = [65, 200, 650]

plt.clf()
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
eta1, = plt.plot([], [], ' ', label="$|\eta| < 1.3$")
ql1, = plt.plot(pt1, QGl_AUC_1, color='red', lw=1, marker='o', markersize=2, label='Likelihood')
fnn1, = plt.plot(pt1, fNN_AUC_1, color='blue', lw=1, marker='o', markersize=2, label='Feedforward')
dj1, = plt.plot(pt1, DJ_AUC_1, color='forestgreen', lw=1, marker='o', markersize=2, label='Sequential Model')
img1, = plt.plot(pt1, Img_AUC_1, color='purple', lw=1, marker='o', markersize=2, label='Jet Image')
eta2, = plt.plot([], [], ' ', label="$1.3 < |\eta| < 2.5$")
ql2, = plt.plot(pt2, QGl_AUC_2, color='red', linestyle='--', lw=1, marker='o', markersize=2, label='Likelihood')
fnn2, = plt.plot(pt2, fNN_AUC_2, color='blue', linestyle=':', marker='o', markersize=2, label='Feedforward')
dj2, = plt.plot(pt2, DJ_AUC_2, color='forestgreen', lw=1, linestyle='--', marker='o', markersize=2, label='Sequential Model')
img2, = plt.plot(pt2, Img_AUC_2, color='purple', linestyle=':', marker='o', markersize=2, label='Jet Image')
plt.xlim([0.0,1050.])
plt.ylim([0.6, 0.9])

# Add legends
legend1 = plt.legend(handles=[eta1,ql1,fnn1,dj1,img1], loc='lower left', framealpha=1)
ax.add_artist(legend1)
legend1.get_frame().set_facecolor('white')
legend1.get_frame().set_linewidth(0.2)
legend2 = plt.legend(handles=[eta2,ql2,fnn2,dj2,img2], loc='lower right', framealpha=1)
ax.add_artist(legend2)
legend2.get_frame().set_facecolor('white')
legend2.get_frame().set_linewidth(0.2)

# Add grid
ax.xaxis.set_major_locator(plt.MultipleLocator(200))
ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
plt.grid(color='grey', which='major', linestyle='--', linewidth=0.4)
plt.grid(color='grey', which='minor', linestyle=':', linewidth=0.1, alpha=0.5)


plt.ylabel('ROC$\;$ AUC')
plt.xlabel('$p_T$')
plt.savefig(outFolder + 'compareAUC_plot.pdf')
