from contextlib import closing
import data_file
import numpy as np
import cv2
from camera_stuff import find_template
import time
import os
import random
from matplotlib import pyplot as plt

if __name__ == "__main__":

    with closing(data_file.Datafile(filename = 'tracking_analysis.hdf5')) as df:

        tracking_analysis_data = df.new_group('data', 'Characterises the amount of white noise the tracking algorithm is able to withstand')
        num = 5
        data = np.zeros((num, 3))
        sig_step = 1
        n_av = 5
        template = cv2.imread(os.path.join('calibration', 'drift_templ8.jpg'))
        temp1ate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        frame = cv2.imread(os.path.join('images', 'drift_20190116_204836.jpg'))
        frame = cv2.imread(os.path.join('images', 'drift_20190116_210704.jpg'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_w, frame_h = frame.shape[::-1]
        orig_pos, corr = find_template(template, frame - np.mean(frame), return_corr = True, fraction = 0.3)
        orig_pos = np.asarray(orig_pos)
        print(orig_pos)
        for i in range(1,num+1):
            sigma = i*sig_step
            data[i-1, 0] = sigma
            pos_change = np.zeros(2)
            for j in range(n_av):
                noise = np.round(sigma*np.random.randn(frame_h,frame_w))
                #change to normal integer to stop wrapping
                noisy_frame = frame.astype(int)+noise
                #Round negative pixels and pixels above 255
                noisy_frame[noisy_frame<0] = 0
                noisy_frame[noisy_frame>255] = 255
                noisy_frame = noisy_frame.astype(np.uint8)
                #print(frame)
                #cv2.imwrite(os.path.join('tracking_analysis', 'frame_%s.jpg' % time.strftime('%Y%m%d_%H%M%S')), frame)
                
                pos, corr = find_template(template, noisy_frame - np.mean(noisy_frame), return_corr = True, fraction = 0.3)
                #Total position change for all images with same noise level, divide by n_av later
                pos_change += pos-orig_pos
            pos_change = pos_change/n_av
            print(np.linalg.norm(pos_change))
            #cv2.imwrite(os.path.join('tracking_analysis', 'corr_%s.jpg' % time.strftime('%Y%m%d_%H%M%S')), corr * 255.0 / np.max(corr))
        df.add_data(data, tracking_analysis_data, 'data')

    calframe = cv2.imread(os.path.join('calibration', 'drift_image.jpg'))
    calframe = cv2.cvtColor(calframe, cv2.COLOR_BGR2GRAY)

    diff_im = calframe.astype(int)-frame
    vals,edges = np.histogram(diff_im.flatten(),30)
    
    cents = (edges[1:]+edges[0:-1])/2
    print(vals,cents)
    #plt.plot(cents,vals)
    plt.imshow(diff_im)
    plt.show()
    #print(frame)

    #print(calframe)