# %%
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import datetime
import glob
import cv2
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
img = np.array(df)
img -= img.mean()
img = np.abs(img)
low, high = np.percentile(img, [3, 99])
img = np.clip(img, low, high)
img -= img.min()
img *= 255.0 / img.max()
img = img.astype(np.uint8)
img

# %%
def imshow(img, lines=[]):
    fig = plt.figure(figsize=(12,16))
    ax = plt.axes()

    norm = Normalize(vmin=0, vmax=255, clip=True)

    im = ax.imshow(img,interpolation='none',aspect='auto',norm=norm)

    x_values = np.linspace(0, img.shape[1], num=100)
    for intercept, slope in lines:
        ls_y_values = intercept + slope * x_values
        ax.plot(x_values, np.clip(ls_y_values, 0, img.shape[0]), color="red", linewidth=2)

    plt.ylabel('time')
    plt.xlabel('space [m]')

    cax = fig.add_axes([ax.get_position().x1+0.06,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    plt.show()

imshow(img)

# %%
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
f = np.fft.fft2(img_grayscale)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# %%
slice_idx = int(0.27 * img.shape[0])
slice = img[slice_idx]

slice

# %%
import cv2

def cvconv(f, g):
    # padding
    pad_v = (g.shape[0] - 1) // 2
    pad_h = (g.shape[1] - 1) // 2
    fb = cv2.copyMakeBorder(f, pad_v, pad_v, pad_h, pad_h, cv2.BORDER_CONSTANT, 0)

    g = np.flip(g)

    # convolution
    fg_cv = cv2.filter2D(fb.astype(g.dtype), -1, g)

    # remove padding from result (opencv does not do this automatically)
    return fg_cv[pad_v : fb.shape[0] - pad_v, pad_h : fb.shape[1] - pad_h]

der = np.array([[-1, 1]], np.float32)

x = np.arange(0, img.shape[1])

y_hat = cvconv(slice[np.newaxis], der) / (x[1] - x[0])
y_hat = y_hat[0]

where = abs(y_hat) < 0.0003
extremes = x[where]
extremes

# %%
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

a = extremes.reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
s = np.linspace(a.min(), a.max())
e = kde.score_samples(s.reshape(-1,1))

mi = argrelextrema(e, np.less)[0]

segments = [a[(a >= left) * (a <= right)] for left, right in zip((s[0],) + tuple(s[mi]), tuple(s[mi]) + (s[-1],))]
segments

# %%
suspected_peaks = [segment.mean() for segment in segments]
suspected_peaks

# %%
def get_peaks(slice):
    der = np.array([[-1, 1]], np.float32)

    x = np.arange(0, img.shape[1])

    y_hat = cvconv(slice[np.newaxis], der) / (x[1] - x[0])
    y_hat = y_hat[0]

    where = abs(y_hat) < 0.0003
    extremes = x[where]

    if len(extremes) == 0:
        return np.array([])

    a = extremes.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
    s = np.linspace(a.min(), a.max())
    e = kde.score_samples(s.reshape(-1,1))

    mi = argrelextrema(e, np.less)[0]

    segments = [a[(a >= left) * (a <= right)] for left, right in zip((s[0],) + tuple(s[mi]), tuple(s[mi]) + (s[-1],))]

    return np.array([segment.mean() for segment in segments])

for i in range(10):
    print(i, [int(x) for x in get_peaks(img[slice_idx+i])])

# %%
from tqdm import trange

# this is just for visualization
peaks_img = np.zeros_like(img)
for i in trange(img.shape[0]):
    for x in get_peaks(img[i]):
        peaks_img[i, int(x)] = 255

imshow(peaks_img)

# %%
def get_peaks(slice, threshold): #min, max):
    der = np.array([[-1, 1]], np.float32)

    x = np.arange(0, img.shape[1])

    y_hat = cvconv(slice[np.newaxis], der) / (x[1] - x[0])
    y_hat = y_hat[0]

    where = abs(y_hat) < 0.0003
    extremes = x[where]

    extreme_values = slice[where]
    extremes = extremes[extreme_values > threshold]
    # extremes = extremes[np.all((extreme_values >= min) & (extreme_values <= max))]

    # if extremes.shape[0] == 0 or extremes.shape[1] == 0:
    if len(extremes) == 0:
        return np.array([])

    a = extremes.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
    s = np.linspace(a.min(), a.max())
    e = kde.score_samples(s.reshape(-1,1))

    mi = argrelextrema(e, np.less)[0]

    segments = [a[(a >= left) * (a <= right)] for left, right in zip((s[0],) + tuple(s[mi]), tuple(s[mi]) + (s[-1],))]

    return np.array([segment.mean() for segment in segments])

peaks_img = np.zeros_like(img)
for i in trange(img.shape[0]):
    for x in get_peaks(img[i], 50):#30, 255):
        peaks_img[i, int(x)] = 255

imshow(peaks_img)

# %%
img_blur = cv2.blur(img, (5, 5))
peaks_img = np.zeros_like(img)
for i in trange(img.shape[0]):
    for x in get_peaks(img_blur[i], 50):#30, 255):
        peaks_img[i, int(x)] = 255

imshow(peaks_img)

# %%
kernel = np.ones((5, 5), np.uint8)

# Proper closing: Close -> Open -> Close
img_proper_closing = cv2.morphologyEx(
    cv2.morphologyEx(
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel),
        cv2.MORPH_OPEN, kernel),
    cv2.MORPH_CLOSE, kernel)

peaks_img = np.zeros_like(img)
for i in trange(img.shape[0]):
    for x in get_peaks(img_proper_closing[i], 70):#30, 255):
        peaks_img[i, int(x)] = 255

imshow(peaks_img)

# %%
current_peak = get_peaks(img[slice_idx])
for i in range(1, 10):
    next_peak = get_peaks(img[slice_idx+i])

    print(current_peak, next_peak)

    # # closest matches
    # max_matches = min(len(last_peak), len(current_peak))
    # # print(max_matches)
    #
    # print(last_peak, current_peak)
    # distances = np.abs(last_peak[:, np.newaxis] - current_peak)
    # # print(distances)
    #
    # distances = distances.min(axis=0)
    # # print(distances)
    #
    # if len(distances) <= max_matches:
    #     indices = np.arange(max_matches)
    # else:
    #     indices = np.argpartition(distances, max_matches)[:max_matches]
    # # print(indices)
    #
    # print(current_peak[indices])

    # print("====")

    current_peak = next_peak

# %%
A = np.array([1, 2, 3])         # Length 3
B = np.array([4, 5, 6])   # Length 5

# Calculate distances using broadcasting
distances = np.abs(A[:, np.newaxis] - B)
print("Distance Matrix (NumPy):")
print(distances)

np.argpartition(distances.min(axis=0), 2)[:2]

# %%
grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
imshow(grad_x)

# %%
grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
imshow(grad)

# %%
_, grad_thold = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow(grad_thold)

# %%
grad_blur = cv2.GaussianBlur(grad, (5,5), 0)
_, grad_blur_thold = cv2.threshold(grad_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow(grad_blur_thold)

# %%
grad_blur_thold_edges = cv2.Canny(grad_blur_thold, 100, 200)
imshow(grad_blur_thold_edges)

# %%
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

grad_blur_thold_skeleton = skeletonize(grad_blur_thold)
grad_blur_thold_skeleton = (grad_blur_thold_skeleton * 255).astype(np.uint8)
imshow(grad_blur_thold_skeleton)

# %%
t = grad_blur_thold_skeleton.copy()
t[img <= 70] = 0
imshow(t)

# %%

