import cv2
import numpy
import utils

def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize = edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""
    def __init__(self, kernel):
        self._kernel = kernel
    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src, -1, self._kernel, dst)

class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)

# look the weights sum up to 1
# we should do this whenever we want to leave
# the image's overall brightness unchanged
# if we modify a sharpening kernel 
# slightly so that its wights sum up to 0 
# instead, we will have an edge detection
# kernel that turns edges white and 
# non-edges black
# for example let's add the following 
# edge detection filter

class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)

# let's now do a blur filter
# for a blur effect the weights should sum up
# to 1 and should be positive throughout
# the neighborhood

class BlurFilter(VConvolutionFilter):
    """A blur filter with a 2-pixel radius."""
    def __init__(self):
        kernel = numpy.array([
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04]
        ])
        VConvolutionFilter.__init__(self, kernel)

# until now we just saw highly symmetric
# kernels, but sometimes kernels with less
# symmetry can be interest
# let's consider a kernel that blurs on one
# side (whit positive weights) and sharpens
# on the other (with negative weights)
# it will produce a ridge or embossed effect
# here is the implementation

class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""
    def __init__(self):
        kernel = numpy.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [ 0, 1, 2]
        ])
        VConvolutionFilter.__init__(self, kernel)