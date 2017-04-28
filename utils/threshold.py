##Refer: :Lecture notes on Advanced Lane Finding
import cv2
import numpy as np

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(25, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

    # Return the binary image
    return binary_output


def combine_thresh(img, color=False, dir_mag_thresh=False):
    
    #combine gradient threshold
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(25, 255))
    dir_binary = dir_threshold(img, sobel_kernel=3, dir_thresh=(0.7, 1.1))
    
    combined_grad = np.zeros_like(dir_binary)
    combined_grad[((mag_binary == 1) & (dir_binary == 1))] = 1

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    r_channel = img[:,:,0]
    
    
    # Threshold color channel
    s_thresh, r_thresh = (170,255), (220, 255)
    s_binary = np.zeros_like(s_channel)
    r_binary = np.zeros_like(r_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    r_binary[(r_channel >= r_thresh[0]) & (s_channel <= r_thresh[1])] = 1
    

    combined_sr = np.zeros_like(s_binary)
    combined_sr[((r_binary == 1) | (s_binary == 1))] = 1
  

    combined_binary = np.zeros_like(mag_binary)
    if dir_mag_thresh:
        combined_binary[((mag_binary ==1 ) & (dir_binary ==1))] = 1
        
    if color:
        return np.dstack(( np.zeros_like(combined_binary), combined_grad, combined_sr))
        
    else:
        combined_sr[((combined_binary ==1) & (combined_sr ==1))] = 1
        return combined_sr