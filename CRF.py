import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np

path = "/home/dpakhom1/dense_crf_python/"
sys.path.append(path)
eps = 1e-5

import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary

import skimage.io as io

for i in range(21):  # Number of image is 25
    '''
    img_init = cv2.imread("CRF_image_" + str(i) + ".jpg", cv2.IMREAD_COLOR)
    img_yuv = cv2.cvtColor(img_init, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    '''
    image = cv2.imread(str(0)+"_"+str(i) + "_0.jpg", cv2.IMREAD_COLOR)
    # image = image[0:200, 0:200, :]

    row = np.shape(image)[0]
    col = np.shape(image)[1]

    softmax = cv2.imread(str(0)+"_"+str(i) + "_1.jpg", cv2.IMREAD_GRAYSCALE)
    # softmax = softmax[0:200, 0:200]
    softmax = softmax / 255.0
    prob = np.zeros([row, col, 2])
    for i in range(row):
        for j in range(col):
            prob[i, j, 0] = min(1-eps, softmax[i, j] + eps)
            prob[i, j, 1] = max(eps, 1 - softmax[i, j] - eps)

    processed_probabilities = prob.transpose((2, 0, 1))
    # print(processed_probabilities)

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = softmax_to_unary(processed_probabilities)
    # print(unary)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(5, 5), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=2,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # dcrf.DenseCRF2D.addPairwiseGaussian()

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    print(np.shape(image))
    feats = create_pairwise_bilateral(sdims=(48, 48), schan=(7, 7, 7),
                                      img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=7,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    for j in range(6):
        Q = d.inference(6 * j)
        # print(np.shape(Q))

        res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

        cv2.imshow("tmp_win1", image)
        cv2.waitKey(0)
        cv2.imshow("tmp_win2", (1 - res) * 255.0)
        cv2.waitKey(0)
        '''
        cmap = plt.get_cmap('bwr')

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
        ax1.set_title('Segmentation with CRF post-processing')
        ax2.imshow(image)
        plt.show()
        '''
