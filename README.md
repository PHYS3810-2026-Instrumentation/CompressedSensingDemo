# CompressedSensingDemo

Requires:
 - numpy
 - scipy
 - matplotlib
 - cvxpy

Reconstructing an image can be done with the functions in `sensing.py`, which assumes RGB (24 bit) images. All three RGB 'frames' are reconstructed independently before being realigned for an RGB image.