import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from src.settings import load_pickle
import ffmpeg



# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
#line, = ax.plot([], [], lw=2)
a=np.random.random((5,5))
im=plt.imshow(a,interpolation='none')
# initialization function: plot the background of each frame
def init():
    original = load_pickle("C:/Users/NN/AppData/Local/Temp/animation/target")
    im.set_data(np.random.random((5,5)))
    return [im]

# animation function.  This is called sequentially


def animate(i):
    a=im.get_array()
    a=a*np.exp(-0.001*i)    # exponential decay of the values
    im.set_array(a)
    return [im]


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()

# def animate_plane(predictions,target):
