"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Runs toy_sine() for various arguments.
"""
import numpy as np
from toy_sine import toy_sine

line_or_sine = "sine"
cyclic = False
Ns = np.array([2])

# for line_or_sine in "sine", "line":
#     for cyclic in False, True:
#         for Ns in np.array([2]), np.array([2, 3, 4, 5, 6, 7]):

toy_sine(line_or_sine, Ns, cyclic, x_errors=False, read_resume=False)
