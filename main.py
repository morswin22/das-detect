# %%
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import datetime
import glob
from matplotlib.colors import Normalize


def set_axis(x, no_labels = 7)->tuple[np.array, np.array]:
    """Sets the x-axis positions and labels for a plot.

    Args:
        x (np.array): The x-axis data.
        no_labels (int, optional): The number of labels to display. Defaults to 7.

    Returns:
        tuple[np.array, np.array]: A tuple containing:
            - The positions of the labels on the x-axis.
            - The labels themselves.
    """
    nx = x.shape[0]
    step_x = int(nx / (no_labels - 1))
    x_positions = np.arange(0,nx,step_x)
    x_labels = x[::step_x]
    return x_positions, x_labels

# %%
DX = 5.106500953873407
DT = 0.0016

FILE_START = "091052"
FILE_END   = "091242"

# %%
path_out = 'data/'
files = glob.glob(path_out+"*")
files.sort()

start_idx = files.index(f"{path_out}{FILE_START}.npy")
end_idx = files.index(f"{path_out}{FILE_END}.npy")

data = []
first_filename = files[start_idx]
for file in files[start_idx:end_idx+1]:
   data.append(np.load(file))
data = np.concatenate(data)
time_start = datetime.datetime.strptime('2024-05-07 ' + first_filename[len(path_out):].split(".")[0], "%Y-%m-%d %H%M%S")
index = pd.date_range(start=time_start, periods=len(data), freq=f'{DT}s')

columns = np.arange(len(data[0])) * DX

df = pd.DataFrame(data=data, index=index, columns=columns)
df

# %%
fig = plt.figure(figsize=(12,16))
ax = plt.axes()

# This is an example transformation and should be converted to the proper algorithm
df -= df.mean()
df = np.abs(df)
low, high = np.percentile(df, [3, 99])
norm = Normalize(vmin=low, vmax=high, clip=True)

im = ax.imshow(df,interpolation='none',aspect='auto',norm=norm)
plt.ylabel('time')
plt.xlabel('space [m]')

cax = fig.add_axes([ax.get_position().x1+0.06,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(im, cax=cax)
x_positions, x_labels = set_axis(df.columns)
ax.set_xticks(x_positions, np.round(x_labels))
y_positions, y_labels = set_axis(df.index.time)
ax.set_yticks(y_positions, y_labels)
plt.show()

# %%
from matplotlib.backend_bases import RendererBase

img_rgba, offset_x, offset_y, transform = im.make_image(RendererBase)
img_rgba

# %%
import cv2
# skimage, PIL

def imshow(image):
    cv2.imshow('ImageWindow', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
imshow(img)

# %%
# grayscale
img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
imshow(img)

# %%
plt.hist(img.reshape(-1), bins=50)
plt.show()

# %%
img[img <= 70] = 0
imshow(img)

# %%
def proper_closing(img, struct):
    img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, struct)
    img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, struct)
    img_close2 = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, struct)
    return np.minimum(img, img_close2)

struct = np.ones([5, 5], np.uint8)
img_Q = proper_closing(img, struct)

imshow(np.concatenate([img, img_Q], 1))

# %%
img_binary = img_Q.copy()
img_binary[img_binary.nonzero()] = 1
img_binary = img_binary.astype(bool)

from scipy.ndimage import binary_hit_or_miss

def thinning(img, struct):
    hit, miss = struct.copy(), struct.copy()
    hit[struct == -1] = 0
    miss[struct == 1] = 0
    miss[struct == -1] = 1
    return np.logical_and(img, np.logical_not(binary_hit_or_miss(img, hit, miss)))

def skeletonize(img):
    s1 = np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]])
    s2 = np.array([[0, -1, -1], [1, 1, -1], [0, 1, 0]])

    result = img.copy()
    while True:
      temp = result.copy()
      for i in range(4):
        temp = np.rot90(thinning(thinning(temp, s1), s2))
      if np.all(temp == result):
        break
      result = temp.copy()

    return result


skeleton = skeletonize(img_binary)

def show_binary(a):
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        labelleft=False,
        left=False,
        right=False,
    )
    plt.imshow(a, cmap="gray")
    plt.show()

show_binary(skeleton)

# %%
skeleton_img = skeleton.astype(np.int8)
imshow(skeleton_img) # TODO how to go back from skeleton to cv2?

# %%
skeleton_Q = proper_closing(skeleton.astype(np.int8) * 255, struct)
imshow(skeleton_Q)
skeleton_Q
# %%

