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

img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
imshow(img_bgr)

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
img_Q_blurred = cv2.blur(img_Q, (10, 10))

imshow(np.concatenate([img_Q, img_Q_blurred], 1))

# %%
img_binary = img_Q_blurred.copy()
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
skeleton_img = (skeleton * 255).astype(np.uint8)
imshow(skeleton_img)

# %%
descriptor = cv2.ORB_create()

kp, desc = descriptor.detectAndCompute(skeleton_img, None)

imshow(cv2.drawKeypoints(skeleton_img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
imshow(cv2.drawKeypoints(img_bgr, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))

# %%
descriptor = cv2.SIFT_create()

kp, desc = descriptor.detectAndCompute(skeleton_img, None)

imshow(cv2.drawKeypoints(skeleton_img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
imshow(cv2.drawKeypoints(img_bgr, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))

# %%
descriptor = cv2.FastFeatureDetector_create()

kp = descriptor.detect(skeleton_img, None)

imshow(cv2.drawKeypoints(skeleton_img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))
imshow(cv2.drawKeypoints(img_bgr, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))

# %%
# Data
kp_data = {"x":[], "y":[]}
for key_point in kp:
    kp_data["x"].append(key_point.pt[0])
    kp_data["y"].append(key_point.pt[1])

# Convert data to DataFrame for easier manipulation
kp_df = pd.DataFrame(kp_data)

# %%
from scipy import stats

fig, axs = plt.subplots(1, 2, sharex=True, tight_layout=True)

axs[0].hist(kp_df["y"], bins=50)

x = np.arange(kp_df["y"].min(), kp_df["y"].max(), 0.1)
y = stats.gaussian_kde(kp_df["y"])(x)
axs[1].plot(x, y)

plt.show()

# %%
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

y_hat = cvconv(y[np.newaxis], der) / (x[1] - x[0])
y_hat = y_hat[0]

where = abs(y_hat) < 0.000000003
extremes = x[where]
extremes

# %%
extremes = extremes[y_hat[where] <= 0]
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
initial_guess_ls = np.array([[segment.mean(), 0] for segment in segments]).reshape(-1)
initial_guess_ls

# %%
from scipy.optimize import minimize

# Parameters
num_lines = len(segments)

# Define the objective function for Least Squares Regression with multiple lines
def least_squares(params):
    a = params[:2 * num_lines].reshape(num_lines, -1)   # Coefficients for each line
    residuals = []

    for i in range(len(kp_df)):
        line_fit = np.array([a[j][0] + a[j][1] * kp_df['x'][i] for j in range(num_lines)])
        residuals.append(np.min(np.abs(kp_df['y'][i] - line_fit)))   # Min residual among lines

    return np.sum(np.square(residuals))

# Optimize using minimize from scipy
result_ls = minimize(least_squares, initial_guess_ls)
coefficients_ls = result_ls.x.reshape(num_lines, -1)

# Plotting the results
plt.figure(figsize=(9,12))
plt.scatter(kp_df['x'], kp_df['y'], color='blue', label='Data Points')

# Plotting Least Squares Lines
x_values = np.linspace(kp_df['x'].min(), kp_df['x'].max(), num=100)
for i in range(num_lines):
    ls_y_values = coefficients_ls[i][0] + coefficients_ls[i][1] * x_values
    plt.plot(x_values, ls_y_values, label=f'Least Squares Line {i+1}', linewidth=2)

# Adding labels and title
plt.title('Multiple-Line Regression: Least Squares')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.ylim(kp_df['y'].min() - 1, kp_df['y'].max() + 1)
plt.legend()
plt.grid()
plt.show()

