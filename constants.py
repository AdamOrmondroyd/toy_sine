"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Constant values used throughout.
"""

line_or_sine = "line"
# line_or_sine = "sine"
n_points = 100
amplitude = 1.0
wavelength = 1.0
x_errors = True
if x_errors:
    sigma_x = 0.05
else:
    sigma_x = 0.0
sigma_y = 0.05
