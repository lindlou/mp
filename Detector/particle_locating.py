import glob as glob
import numpy as np
import trackpy as tp
import pims
from skimage.filters import difference_of_gaussians, median, gaussian
import matplotlib.pyplot as plt

@pims.pipeline  # here we are defining a pipeline that will convert an RGB movie frame into a grayscale image.
def as_grey(frame):
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    return 0.05 * red + 0.99 * green + 0.05 * blue


@pims.pipeline  # substracts the background for each movie frame
def bgd_sub(frame, bgframe):
    return np.absolute(frame - bgframe)


@pims.pipeline
def blur_stack(frame):
    return difference_of_gaussians(frame, 0.2, 10)
    # return median(frame)


@pims.pipeline
def gauss_blur(frame, sigma=5):
    return gaussian(frame, sigma)


def calc_bgframe(frames, n=10):
    path = "/Shared/analysis_did_feb2023/particle_tracking" # specify the path where your videos are saved
    # Calculate the background by taking the median frame of n frames evenly distributed in the stack
    ind = np.arange(0, len(frames), np.round(len(frames) / n), dtype=int)
    # Calculate the background using the median
    substack = frames[ind]
    bgd = np.median(substack, axis=0)
    return bgd


def bg_sliding_window_numpy(frames, n=2000):
    """
    Calculate the background by taking the median of n frames locally.
    Substract background frame from original image
    """
    bg_subs = np.zeros(np.shape(frames))
    last_bg = np.median(frames[-n:], axis=0)
    for k in range(0, len(frames) - n):
        ind = np.arange(k, n)
        # Calculate the background using the median
        substack = frames[ind]
        bgd = np.median(substack, axis=0)
        bg_subs[k] = frames[k] - bgd
    bg_subs[-n-1:] = frames[-n-1:] - last_bg
    return bg_subs

def bg_sliding_window(video, n=2000):
    """
    Calculate the background by taking the median of n frames locally.
    Substract background frame from original image
    """
    bg_subs = pims.Frame(np.zeros([len(video),video[0].shape[0],video[0].shape[1]]))
    last_bg = np.median(video[-n:-1], axis=0)#for some reason I get 2 ind less
    for k in range(0, len(video)-n, n):
        # Calculate the backgrounds using the median
        substack = video[k:k+n]
        bgd = np.median(substack, axis=0)
        bg_subs[k:k+n] =np.abs(video[k:k+n]-bgd)
    print(k)
    bg_subs[-n :-1] = np.abs(video[-n :-1] - last_bg)
    return bg_subs


# define path to files to track
path = "/Shared/analysis_did_feb2023/particle_tracking/"# specify the path where your videos are saved
#files = glob.glob(path + '*.mp4')
#files.sort()
#file = files[0]
file = "1Param1_3.mp4"
video = as_grey(pims.open(file))
bgd = calc_bgframe(video, n=2000)
bgd_subs = bgd_sub(video, bgd)#remove background
# #bgd_subs=gauss_blur(bgd_subs,sigma=3)#Apply gaussian blur --> this deforms the cells so we are not using this
bgd_subs = video
sigma=1
n=2000
im=bg_sliding_window(video, n=n)
# #bgd_subs=gauss_blur(im,sigma=sigma)#Apply gaussian blur --> this deforms the cells so we are not using it
#
#%% This part is to save the preprocessed video
# from skimage import io
# import imageio
# im = io.imread(video)
# imageio.mimwrite(file[:-4]+'_preprocessed.mp4', im, fps=30, macro_block_size=1)

#%%
# #plot maximum projection of video + trajectory
# savgol = np.load(path + '1did10param1_1cm_preprocessed_filtered.npy')
# im_max = np.max(video, axis=0)
# plt.imshow(im_max)
# plt.plot(savgol[:,0], savgol[:, 1], linewidth=0.8)
# plt.xlim([0,800])
#  plt.ylim([-20,600])
# plt.title(file[:8])
# # plt.savefig(path + file[:8] + '_maxprojection+TPsavgol5_maxdisp50.png')
# plt.show()

#%%
# Now that we have a preprocessed video we will first identify cells. First analyze a few movie frames individually
#the values chosen for diameter and minmass will change the output
diameter=15
minmass=1500
frame = 400
test = tp.locate(im[frame], diameter=diameter, minmass=minmass, preprocess="False")
tp.annotate(test, im[frame])
#%%
# Once you have determined which parameters work the best you can process several frames from the video. If your
# video is long this can take a while.

f = tp.batch(im, processes=0, diameter=diameter, minmass=minmass,
                 preprocess="False")  # It is key to have the parameter processes set to zero or this freaks out


np.savetxt(
    file[0:-4] + "_positions.txt", f, fmt="%10.2f", delimiter=" ", newline="\n"
)

# Now we will save the pandas data frame with all particle positions for further analysis
f.to_pickle(file[0:-4] + "_particles.pkl")

partfind_params={}
partfind_params["nwindow"]=n
partfind_params["sigma"]=sigma
partfind_params["diameter"]=diameter
partfind_params["minmass"]=minmass  # list of parameter names, will be used later to store settings

savepath=file[:-4]+'_locate_params.txt'
with open(savepath, 'w') as data:  # save dictionary to text file
    data.write(str(partfind_params))
# Now we will save the pandas data frame with all particle positions for further analysis
f.to_pickle(file[0:-4] + "_particles.pkl")

partfind_params={}
partfind_params["nwindow"]=n
partfind_params["sigma"]=sigma
partfind_params["diameter"]=diameter
partfind_params["minmass"]=minmass  # list of parameter names, will be used later to store settings

savepath=file[:-4]+'_locate_params.txt'
with open(savepath, 'w') as data:  # save dictionary to text file
    data.write(str(partfind_params))