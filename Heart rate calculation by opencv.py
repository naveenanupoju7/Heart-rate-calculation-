#importing all the required packages

import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import pi
import scipy.fftpack as sf
import scipy.signal as sig
plt.rcParams['figure.figsize']=[16,12]
plt.rcParams.update({'font.size':18})

#importing the cascade classifier for detecting of face


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
#cap = cv2.VideoCapture(0)
# To use a video file as input 
cap = cv2.VideoCapture('video1.mp4')

#a list to store all the pixels values at forehead at different frames
region=[]
#variable a to count the number of frames
n=0
#running the loop to detect the face and setting the boundary for forehead and calculating the average value of pixels at each frame
while True:
    n=n+1
    
    check,frame=cap.read()
    
    #cv2.imshow('capturing',frame)
    #key=cv2.waitKey(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #Draw the rectagle around forehead    
    cv2.rectangle(frame,(670,190),(730,250),(0,255,0),2)    

    cv2.imshow('img',frame)
    
    im_cropped = frame[190:250,670:730]
    g=im_cropped[:,:,1]
    
    region.append(np.mean(g))
    
    cv2.imshow("Cropped Image", g)
    key=cv2.waitKey(1)

    if n==850:
        break
import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(region)
plt.ylabel('average green pixel value at forehead')
plt.show()

#take spectral analysis

#Fast Fourier Transform(FFT)
n=500
fhat =np.fft.fft(region,n)#compute FFT
PSD=fhat*np.conj(fhat)/n #power spectral density

freq=(1/(n))*np.arange(n)# createx-axis o frequenicies
L=np.arange(1,np.floor(n/2),dtype='int')#only plot the first half


plt.plot(freq[L],PSD[L],color='c',linewidth=1.5,label='Noisy')
plt.xlim(freq[L[0]],freq[L[-1]])
plt.legend()

plt.show()

indices= PSD>5 #find all frequencies with large power
PSDclean =PSD *indices#zero out all others
fhat=indices*fhat  #zero out small fourier coefficients in Y
ffilt=np.fft.ifft(fhat)# inverse FFT for filtered time signal

plt.plot(ffilt,color='c',linewidth=1.5,label='signal after removing noise')

plt.legend()

plt.show()





