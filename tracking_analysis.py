from contextlib import closing
import data_file
import numpy as np
import cv2
import os
from camera_stuff import find_template
import time
import matplotlib.pyplot as plt
import matplotlib

if __name__ == "__main__":
    with closing(data_file.Datafile(filename = 'tracking_analysis.hdf5')) as df:

        n_sig = 5
        n_av = 5
        sig_step = 1

        tracking_analysis_data = df.new_group('data', 'Characterises the amount of white noise the tracking algorithm is able to withstand')
        data = np.zeros((n_sig, 3))

        template = cv2.imread(os.path.join('calibration', 'drift_templ8.jpg'))
        temp1ate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        frame = cv2.imread(os.path.join('images', 'drift_20190116_204836.jpg'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_w, frame_h = frame.shape[::-1]
        orig_pos, corr = find_template(template, frame - np.mean(frame), return_corr = True, fraction = 0.3)
        orig_pos = np.asarray(orig_pos)
        print(orig_pos)

        for i in range(1, n_sig + 1):
            sigma = i * sig_step
            data[i - 1, 0] = sigma
            pos_change = np.zeros(2)
            for j in range(n_av):
                noise = np.round(sigma * np.random.randn(frame_h, frame_w))
                #change to normal integer to stop wrapping
                noisy_frame = frame.astype(int) + noise
                #Round negative pixels and pixels above 255
                noisy_frame[noisy_frame < 0] = 0
                noisy_frame[noisy_frame > 255] = 255
                noisy_frame = noisy_frame.astype(np.uint8)
                cv2.imwrite(os.path.join('tracking_analysis', 'noisy_frame_%s.jpg' % time.strftime('%Y%m%d_%H%M%S')), noisy_frame)
                pos, corr = find_template(template, noisy_frame - np.mean(noisy_frame), return_corr = True, fraction = 0.3)
                #Total position change for all images with same noise level, divide by n_av later
                pos_change += pos - orig_pos
            pos_change = pos_change / n_av
            data[i - 1, 1:] = pos_change
            print(np.linalg.norm(pos_change))
        df.add_data(data, tracking_analysis_data, 'data')

    calframe = cv2.imread(os.path.join('calibration', 'drift_image.jpg'))
    calframe = cv2.cvtColor(calframe, cv2.COLOR_BGR2GRAY)

    diff_im = calframe.astype(int) - frame

    matplotlib.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots(1, 1)

    ax.plot(data[:, 0], data[:, 1], 'ro-')
    ax.plot(data[:, 0], data[:, 2], 'bo-')

    ax.set_xlabel('Sigma')
    ax.set_ylabel(r'Change in position [$\mathrm{px}$]')

    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots(1, 1)
    hist, bins = np.histogram(diff_im.flatten(), 30)

    ax1.plot(bins[1:], hist)

    ax1.set_xlabel('Difference between frame and calibration frame')
    ax1.set_ylabel('Counts')

    plt.tight_layout()
    plt.show()
