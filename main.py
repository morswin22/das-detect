# %%
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import datetime
import glob
import cv2
from matplotlib.colors import Normalize
from scipy.integrate import quad # for integrating abs(f(x) - g(x))
from sklearn.cluster import DBSCAN # for clustering lines

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

def calculate_slope_and_intercept(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')  # Handle vertical line
    b = y1 - a * x1
    return a, b

def imshow(img, lines=[], lines_scale=np.ones(shape=(2,)), save=None):
    fig = plt.figure(figsize=(12,16))
    ax = plt.axes()

    norm = Normalize(vmin=0, vmax=255, clip=True)

    im = ax.imshow(img,interpolation='none',aspect='auto',norm=norm)

    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            if l[0] == l[2]: # no movement
                continue
            scaled_xs = l[0::2] * lines_scale[1]
            scaled_ys = l[1::2] * lines_scale[0]
            a, b = calculate_slope_and_intercept(scaled_xs[0], scaled_ys[0], scaled_xs[1], scaled_ys[1])
            speed = float("inf") if a == 0 else 1/abs(a) * DX/DT * 3.6
            if speed < 1 or speed > 200: # nobody would drive this fast or this slow
                continue
            scaled_xs = np.clip(scaled_xs, 0, img.shape[1] - 1)
            scaled_ys = a * scaled_xs + b
            ax.text(scaled_xs[0], scaled_ys[0], f"{round(speed, 2)} km/h", color="red", horizontalalignment="center", verticalalignment="top")
            ax.plot(scaled_xs, scaled_ys, color="red")

    plt.ylabel('time')
    plt.xlabel('space [m]')

    if save is None:
        cax = fig.add_axes([ax.get_position().x1+0.06,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(im, cax=cax)
    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    if save:
        plt.tight_layout()
        fig.savefig(save)
    plt.show()

from itertools import cycle, islice

def imshow_downsampled(img, lines=[], labels=[], save=None):
    aspect = img.shape[1] / img.shape[0]
    fig = plt.figure(figsize=(12 * aspect, 12))
    ax = plt.axes()

    norm = Normalize(vmin=0, vmax=255, clip=True)

    im = ax.imshow(img,interpolation='none',aspect='auto', norm=norm)

    if lines is not None and len(lines) > 0:
        if len(labels) < len(lines):
            labels = (np.ones(shape=(len(lines),)) * -1).astype(int)
        colors = np.array(list(islice(cycle([
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
            "#984ea3",
            "#999999",
            # "#e41a1c",
            "#dede00",
        ]), int(max(labels) + 1),)))
        # add red color for outliers (if any)
        colors = np.append(colors, ["red"])

        for l, label in zip(lines[:, 0], labels):
            ax.plot([l[0], l[2]], [l[1], l[3]], color=colors[label])

    plt.ylabel('time')
    plt.xlabel('space')

    if save is None:
        cax = fig.add_axes([ax.get_position().x1+0.06,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(im, cax=cax)
    if save:
        plt.tight_layout()
        fig.savefig(save)
    plt.show()

# %%
DX = 5.106500953873407
DT = 0.0016

# 156053:
FILE_START = "091052"
FILE_END   = "091242"
# 156042:
# FILE_START = "090252"
# FILE_END   = "090442"
# Unassigned:
# FILE_START = "093652"
# FILE_END   = "093842"

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

img = np.array(df)
img -= img.mean()
img = np.abs(img)
low, high = np.percentile(img, [3, 99])
img = np.clip(img, low, high)
img -= img.min()
img *= 255.0 / img.max()
img = img.astype(np.uint8)
imshow(img)

# %%
def masked_norm(fft, mask):
    filtered_fft = fft * mask.astype(int)
    filtered_img = np.fft.ifft2(filtered_fft).real
    norm_img = np.abs(filtered_img - filtered_img.mean())
    thold = np.percentile(norm_img, 99)
    norm_img = np.minimum(norm_img, thold)
    norm_img -= norm_img.min()
    norm_img *= 255.0 / norm_img.max()
    return norm_img.astype(np.uint8)

fft = np.fft.fft2(np.array(df))

frequencies_x = np.fft.fftfreq(img.shape[1], DX)
frequencies_y = np.fft.fftfreq(img.shape[0], DT)

frequencies_x_mesh, frequencies_y_mesh = np.meshgrid(frequencies_x, frequencies_y)
frequencies = np.sqrt(frequencies_x_mesh**2 + frequencies_y_mesh**2)

frequencies.min(), frequencies.max()

# %%
img_norm = masked_norm(fft, (frequencies >= 40) & (frequencies <= 60))
imshow(img_norm, save="filtered_and_normalised.png")

# %%
def create_frequency_mask(shape, freq_min, freq_max):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - center_col)**2 + (Y - center_row)**2)

    mask = np.logical_and(freq_min <= dist_from_center, dist_from_center <= freq_max)
    return mask.astype(np.float32)

fft = np.fft.fft2(img_norm)
fshift = np.fft.fftshift(fft)

mask_1 = create_frequency_mask(img_norm.shape, 1, 32)
mask_2 = create_frequency_mask(img_norm.shape, 1, 64)

mask_1_fshift = mask_1 * fshift
mask_2_fshift = mask_2 * fshift
both_pass_fshift = (mask_1_fshift + mask_2_fshift)

f_ishift = np.fft.ifftshift(both_pass_fshift)
both_pass_fshift_img = np.fft.ifft2(f_ishift).real

img_filtered = both_pass_fshift_img.clip(0, 255).astype(np.uint8)
imshow(img_filtered, save="low_filtered.png")

# %%
img_downsampled = cv2.resize(img_filtered, (256, 256), interpolation=cv2.INTER_AREA)
imshow_downsampled(img_downsampled, save="downsampled.png")

# %%
lines = cv2.HoughLinesP(img_downsampled, rho=1, theta=np.pi/180, threshold=100, lines=None, minLineLength=4, maxLineGap=128)
imshow_downsampled(img_downsampled, lines, save="hough.png")

# %%
def line_function(line):
    """Returns a function representing a line segment defined by (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    def f(x):
        return slope * x + intercept

    return f

def average_distance(line1, line2):
    """Calculates the average distance between two lines over their overlapping interval."""
    if line1[0] == line1[2] or line2[0] == line2[2]:
        return 100000 # No point in clustering these lines

    f1 = line_function(line1)
    f2 = line_function(line2)

    # Determine overlapping interval
    x1_min = min(line1[0], line1[2])
    x1_max = max(line1[0], line1[2])

    x2_min = min(line2[0], line2[2])
    x2_max = max(line2[0], line2[2])

    overlap_min = max(x1_min, x2_min)
    overlap_max = min(x1_max, x2_max)

    if overlap_min >= overlap_max:
        return 100000#float("inf")  # No overlap

    # Define the integrand for the integral of |f(x) - g(x)|
    def integrand(x):
        return abs(f1(x) - f2(x))

    # Calculate integral over the overlapping interval
    integral_value, _ = quad(integrand, overlap_min, overlap_max)

    # Calculate average distance
    average_distance_value = integral_value / (overlap_max - overlap_min)

    return average_distance_value

def compute_distance_matrix(lines):
    """Computes the average distance matrix for a set of lines."""
    n = len(lines)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = average_distance(lines[i], lines[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist  # Symmetric matrix

    return distance_matrix

def cluster_lines(lines, eps=0.5, min_samples=2):
    """Clusters lines using DBSCAN based on their average distances."""
    distance_matrix = compute_distance_matrix(lines)

    # Use DBSCAN with precomputed distances
    clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')

    labels = clustering_model.fit_predict(distance_matrix)

    return labels

labels = cluster_lines(lines[:, 0], eps=4.5, min_samples=2)
imshow_downsampled(img_downsampled, lines, labels, save="clustered.png")

# %%
def merge_lines_in_clusters(lines, labels):
    """Merges lines that are in the same cluster."""
    unique_labels = set(labels)
    merged_lines = []
    merged_lines.extend(lines[labels == -1])

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points

        # Get indices of lines in this cluster
        cluster_lines_indices = np.where(labels == label)[0]

        # Average coordinates of lines in this cluster
        if len(cluster_lines_indices) > 0:
            avg_x1 = np.mean(lines[cluster_lines_indices][:, 0])
            avg_y1 = np.mean(lines[cluster_lines_indices][:, 1])
            avg_x2 = np.mean(lines[cluster_lines_indices][:, 2])
            avg_y2 = np.mean(lines[cluster_lines_indices][:, 3])

            merged_line = [avg_x1, avg_y1, avg_x2, avg_y2]
            merged_lines.append(merged_line)

    return np.array(merged_lines)

merged_lines = merge_lines_in_clusters(lines[:, 0], labels).reshape(-1, 1, 4).astype(int)
imshow_downsampled(img_downsampled, merged_lines, save="merged.png")

# %%
imshow(img, merged_lines, np.array(img.shape) / np.array(img_downsampled.shape), save="result.png")

