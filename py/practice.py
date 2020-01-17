import os, argparse, sys, shutil
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np

x = np.random.randint(10, 50, size=1000)
y = x + np.random.poisson(5, size=1000)
y_sq = np.power(y, 2)

x_range = list(range(x.min(), x.max() + 1))
y_range = list(range(y.min(), y.max() + 1))
y2_range = list(range(y_sq.min(), y_sq.max() + 1))

x_counts, x_bins = np.histogram(x, bins=x_range + [x.max() + 1])
x_distn = x_counts/x_counts.sum() # marginal

xy_joint_distn = np.zeros((len(y_range), len(x_range)), dtype=int)
xy2_joint_distn = np.zeros((len(y2_range), len(x_range)), dtype=int)

for i, x_val in enumerate(x_range):
    val_inds = np.nonzero(x == x_val)[0]
    if val_inds.size > 0:
        y_vals_given_x_val, y_counts_given_x_val = np.unique(y[val_inds], return_counts=True)
        y2_vals_given_x_val, y2_counts_given_x_val = np.unique(y_sq[val_inds], return_counts=True)
        xy_joint_distn[[y_range.index(v) for v in y_vals_given_x_val], i] += y_counts_given_x_val
        xy2_joint_distn[[y2_range.index(v) for v in y2_vals_given_x_val], i] += y2_counts_given_x_val

xy_joint_distn = xy_joint_distn / xy_joint_distn.sum() # joint distn
xy2_joint_distn = xy2_joint_distn / xy2_joint_distn.sum() # joint distn

y_given_x_distn = xy_joint_distn / x_distn # cond distn
y2_given_x_distn = xy2_joint_distn / x_distn # cond distn

exp_y_given_x = np.zeros(len(x_range)) 
exp_y2_given_x = np.zeros(len(x_range))

for i,x_val in enumerate(x_range):
    exp_y_given_x[i] += np.dot(y_range, y_given_x_distn[:,i])
    exp_y2_given_x[i] += np.dot(y2_range, y2_given_x_distn[:,i])

var_y_given_x = exp_y2_given_x - np.power(exp_y_given_x, 2)

exp_var_y_given_x = np.dot(x_distn, var_y_given_x)

var_exp_y_given_x = np.dot(x_distn, np.power(exp_y_given_x, 2)) - np.power(np.dot(x_distn, exp_y_given_x), 2)

np.isclose(np.var(y), exp_var_y_given_x + var_exp_y_given_x) # works

# NB Need to do something similar for two variables

