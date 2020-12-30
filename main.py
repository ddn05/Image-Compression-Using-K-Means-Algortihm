import numpy as np
import cv2 as cv

def K_Means (Image, K) :
    if(len(Image.shape) < 3):
        Z = Image.reshape((-1,1))
    elif (len(Image.shape) == 3 ):
        Z = Image.reshape((-1,3))

    #convert to np.float32
    Z = np.float32(Z)

    #Define Criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    #Apply K_Means
    ret, label, center = cv.kmeans(Z,K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    #Now Convert back the float 32 data to unit 8 and make the image
    center = np.uint8(center)
    res = center[label.flatten()]
    Clustered_Image = res.reshape((Image.shape))

    return Clustered_Image
def main():
    Input_Image = cv.imread("dino.png")

    Clusters = 8

    Clustered_Image = K_Means(Input_Image, Clusters)

    cv.imwrite("Clustered_Image.png", Clustered_Image)
    input("Please enter continue")

if __name__ == '__main__':
    main()
