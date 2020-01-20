import os, argparse, sys, shutil
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np

x = np.random.randint(10, 50, size=1000)
signal = np.random.poisson(5, size=1000) 
noise = np.random.normal(0, 1, size=1000)
x = x + noise
y = x + signal  
xy = x*y

noise_counts, noise_bins = np.histogram(noise, bins=50)
noise_distn = noise_counts/noise_counts.sum()
noise_value_bins = np.digitize(noise, noise_bins) - 1
noise_value_bins[noise_value_bins == noise_counts.size] = noise_counts.size - 1

x_counts, x_bins = np.histogram(x, bins=40)
x_distn = x_counts/x_counts.sum() # marginal
x_bin_centres = x_bins[:-1] + np.diff(x_bins)
x_value_bins = np.digitize(x, x_bins) - 1
x_value_bins[x_value_bins == x_counts.size] = x_counts.size - 1

y_counts, y_bins = np.histogram(y, bins=50)
y_distn = y_counts/y_counts.sum()
y_bin_centres = y_bins[:-1] + np.diff(y_bins)
y_value_bins = np.digitize(y, y_bins) - 1
y_value_bins[y_value_bins == y_counts.size] = y_counts.size - 1

xy_counts, xy_bins = np.histogram(xy, bins=50)
xy_distn = xy_counts/xy_counts.sum()
xy_bin_centres = xy_bins[:-1] + np.diff(xy_bins)
xy_value_bins = np.digitize(xy, xy_bins) - 1 
xy_value_bins[xy_value_bins == xy_counts.size] = xy_counts.size - 1

xz_joint_distn = np.zeros((x_counts.size, noise_counts.size), dtype=int)
yz_joint_distn = np.zeros((y_counts.size, noise_counts.size), dtype=int)
xyz_joint_distn = np.zeros((xy_counts.size, noise_counts.size), dtype=int)

for i in range(noise_counts.size):
    noise_val_inds = np.nonzero(noise_value_bins == i)[0]
    if noise_val_inds.size > 0:
        x_value_bins_given_z, x_counts_given_z = np.unique(x_value_bins[noise_val_inds], return_counts=True)
        y_value_bins_given_z, y_counts_given_z = np.unique(y_value_bins[noise_val_inds], return_counts=True)
        xy_value_bins_given_z, xy_counts_given_z = np.unique(xy_value_bins[noise_val_inds], return_counts=True)
        xz_joint_distn[x_value_bins_given_z, i] += x_counts_given_z
        yz_joint_distn[y_value_bins_given_z, i] += y_counts_given_z
        xyz_joint_distn[xy_value_bins_given_z, i] += xy_counts_given_z

xz_joint_distn = xz_joint_distn / xz_joint_distn.sum() # joint distn
yz_joint_distn = yz_joint_distn / yz_joint_distn.sum() # joint distn
xyz_joint_distn = xyz_joint_distn / xyz_joint_distn.sum() # joint distn

x_given_z_distn = xz_joint_distn / noise_distn # cond distn
y_given_z_distn = yz_joint_distn / noise_distn # cond distn
xy_given_z_distn = xyz_joint_distn / noise_distn # cond distn

x_given_z_distn[np.isnan(x_given_z_distn)] = 0
y_given_z_distn[np.isnan(y_given_z_distn)] = 0
xy_given_z_distn[np.isnan(xy_given_z_distn)] = 0

exp_x_given_z = np.zeros(noise_counts.size) 
exp_y_given_z = np.zeros(noise_counts.size)
exp_xy_given_z = np.zeros(noise_counts.size)

for i in range(noise_counts.size):
    exp_x_given_z[i] += np.dot(x_bin_centres, x_given_z_distn[:, i])
    exp_y_given_z[i] += np.dot(y_bin_centres, y_given_z_distn[:, i])
    exp_xy_given_z[i] += np.dot(xy_bin_centres, xy_given_z_distn[:, i])

exp_product_exp_values = np.dot(noise_distn, exp_x_given_z * exp_y_given_z)

cov_cond_exp_values = exp_product_exp_values - (x.mean() * y.mean())

exp_cond_cov = np.mean(x*y) - np.dot(noise_distn, exp_x_given_z * exp_y_given_z)

np.isclose(np.cov(x, y), cov_cond_exp_values + exp_cond_cov) # works

# NB Need to do something similar for two variables

