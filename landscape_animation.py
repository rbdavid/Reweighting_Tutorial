
# ----------------------------------------
# USAGE:

# ----------------------------------------
# PREAMBLE:

import sys
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

plt.rc('axes', axisbelow=True)

xlim = (-28.5, 20.5)
ylim = (-8.0, 15.0)

x_axis_half_bins = np.loadtxt(sys.argv[1])[:,1]
y_axis_half_bins = np.loadtxt(sys.argv[2])[:,1]
two_d_fe = np.loadtxt(sys.argv[3])

if two_d_fe.shape != (len(x_axis_half_bins),len(y_axis_half_bins)):
    print('Something went wrong ', two_d_fe.shape, ' != ', len(x_axis_half_bins),len(y_axis_half_bins))
    sys.exit()

masked_fe_counts = ma.masked_where(np.isinf(two_d_fe),two_d_fe)

delta_x = x_axis_half_bins[1] - x_axis_half_bins[0]
delta_y = y_axis_half_bins[1] - y_axis_half_bins[0]
xBins = len(x_axis_half_bins)
yBins = len(y_axis_half_bins)
x_edges = np.array([xlim[0] + delta_x*i for i in range(xBins+1)])
y_edges = np.array([ylim[0] + delta_y*i for i in range(yBins+1)])

# first, animating x-axis range for y-axis fe
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8.0,4.0))

fe_surface = ax1.pcolormesh(x_edges,y_edges,masked_fe_counts.T,cmap='Blues_r',zorder=3,vmax=10)
cb1 = fig.colorbar(fe_surface,ax=ax1)
cb1.set_label(r'Relative Free Energy (kcal mol$^{-1}$)',size=14)
step_line = ax1.plot([],[],'r-',zorder=5,alpha=0.25)
ax1.set_xlabel('Projection onto PC1',size=16)
ax1.set_ylabel('Projection onto PC2',size=16)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_aspect('equal')

y_axis_line = ax2.plot([],[],'k-',lw=2)     #,orientation='horizontal'
ax2.set_xlabel('Projection onto PC2',size=16)
#ax2.set_ylabel('Relative Free Energy  (kcal mol$^{-1}$)',size=16)
ax2.set_xlim(ylim)
ax2.set_ylim((-0.1,10))
ax2.grid(b=True,which='major',axis='both',color='#808080',linestyle='--')

plt.tight_layout()

def init():
    return ax1,ax2

def animate(i):
    step_line[0].set_data([x_axis_half_bins[i],x_axis_half_bins[i]],[ylim[0],ylim[1]])
    y_axis_line[0].set_data(y_axis_half_bins,two_d_fe[i,:])
    return ax1, ax2

anim = FuncAnimation(fig,animate,init_func=init,frames=len(x_axis_half_bins),interval=1000,repeat=False,blit=False)   #, fargs=[colors]
anim.save('REUS_results.y_axis.mp4',extra_args=['-vcodec','libx264'],bitrate=5000,dpi=600,savefig_kwargs=dict(transparent=True))
plt.close()

# second, animating y-axis range for x-axis fe
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8.0,4.0))

fe_surface = ax1.pcolormesh(x_edges,y_edges,masked_fe_counts.T,cmap='Blues_r',zorder=3,vmax=10)
cb1 = fig.colorbar(fe_surface,ax=ax1)
cb1.set_label(r'Relative Free Energy (kcal mol$^{-1}$)',size=14)
step_line = ax1.plot([],[],'r-',zorder=5,alpha=0.25)
ax1.set_xlabel('Projection onto PC1',size=16)
ax1.set_ylabel('Projection onto PC2',size=16)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_aspect('equal')

y_axis_line = ax2.plot([],[],'k-',lw=2)     #,orientation='horizontal'
ax2.set_xlabel('Projection onto PC1',size=16)
#ax2.set_ylabel('Relative Free Energy  (kcal mol$^{-1}$)',size=16)
ax2.set_xlim(xlim)
ax2.set_ylim((-0.1,10))
ax2.grid(b=True,which='major',axis='both',color='#808080',linestyle='--')

plt.tight_layout()

def init():
    return ax1,ax2

def animate(i):
    step_line[0].set_data([xlim[0],xlim[1]],[y_axis_half_bins[i],y_axis_half_bins[i]])
    y_axis_line[0].set_data(x_axis_half_bins,two_d_fe[:,i])
    return ax1, ax2

anim = FuncAnimation(fig,animate,init_func=init,frames=len(y_axis_half_bins),interval=1000,repeat=False,blit=False)   #, fargs=[colors]
anim.save('REUS_results.x_axis.mp4',extra_args=['-vcodec','libx264'],bitrate=5000,dpi=600,savefig_kwargs=dict(transparent=True))
plt.close()







