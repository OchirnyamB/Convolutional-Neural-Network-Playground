from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    # grab the spatial dimensions of the image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # Padding so the output image will have the same dimension
    pad = (kW-1)//2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    for y in np.arange(pad, iH+pad):
        for x in np.arange(pad, iW+pad):
            # Extract the region of interest of image by extracting the center region of the current x,y
            roi = image[y-pad:y+pad+1, x-pad:x+pad+1]

            # Perform convolution
            k = (roi*K).sum()

            # Store the convolved value in the output (x,y) coordinate of the output image
            output[y-pad, x-pad] = k

            # Rescale the output image to be in the range [0, 255]
            output = rescale_intensity(output, in_range=(0, 255))
            output = (output*255).astype("uint8")

            return output

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())



# Construct kernels 
smallBlur = np.ones((7,7), dtype="float")*(1.0/(7*7))
largeBlur = np.ones((21,21), dtype="float")*(1.0/(21*21))

sharpen = np.array(([0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]), dtype="int")
# Edge like regions
laplacian = np.array(([0,-1,0],
                    [1,4,1],
                    [0,1,0]), dtype="int")

# Edge like regions along the x and y axis
sobelX = np.array(([-1,0,1],
                    [-2,0,2],
                    [-1,0,1]), dtype="int")

sobelY = np.array(([1,-2,-1],
                   [0,0,0],
                   [1,2,1]), dtype="int")

emboss = np.array(([-2,-1,0],
                   [-1,1,1],
                   [0,1,2]), dtype="int")
kernelBank = (("smallBlur", smallBlur), ("largeBlur", largeBlur),
("sharpen", sharpen),
("laplacian", laplacian),
("sobelX", sobelX),
("sobelY", sobelY),
("emboss", emboss))

# Laod the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, K) in kernelBank:
    # Apply the kernel to the grayscale image using both our custom
    # kernels and OpenCV's filter2D function
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    # Show the output images
    cv2.imshow("Original", gray)
    cv2.imshow("{} = convolve".format(kernelName), convolveOutput)
    cv2.imshow("{} = opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()