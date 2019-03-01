import argparse
import h5py
import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import time

def printProgressBar(iteration, total, length = 10):
    percent = 100.0 * iteration / total
    filledLength = int(length * iteration // total)
    bar = '*' * filledLength + '-' * (length - filledLength)
    print('Progress: |%s| %d%% Completed' % (bar, percent), end = '\r')
    if iteration == total: 
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive plot of drift data that shows ralavent drift and correlation images upon clicking on a data point")
    parser.add_argument("axis", type = str, help="Plot X or Y position coordinates?")
    args = parser.parse_args()

    axis = args.axis

    print ("Loading data...")

    N_frames = 1000
    #need to be consistant between drift.py and drift_plot.py

    df = h5py.File("drift.hdf5", mode = "r")
    group = list(df.values())[-1]
    N_points = len(group) - 2
    data = np.zeros([N_frames * N_points, 5])
    for i in range(N_points):
        dset = group["data%05d" % i]
        for j in range(N_frames):
            data[N_frames * i + j, :] = dset[j, :]
        printProgressBar(i, N_points)
    print("")

    image_path = r"C:\Users\qm237\Documents\PhD\Christmas drift\images"
    N_images = len(os.listdir(image_path))

    original_image = np.empty((N_images//2, 480, 640))
    corr_image = np.empty((N_images//2, 289, 289, 3))
    image_files = [filename for filename in os.listdir(image_path) if filename.startswith("corr")]
    for i in range(N_images//2):
        corr_image[i, :, :, :] = cv2.imread(os.path.join('images', image_files[i]), 1)
    image_files = [filename for filename in os.listdir(image_path) if filename.startswith("drift")]
    for j in range(N_images//2):
        original_image[j, :, :] = cv2.imread(os.path.join('images', image_files[i]), 0)

    matplotlib.rcParams.update({'font.size': 12})

    microns_per_pixel = 2.74

    t = data[:, 0]
    humidity = data[:, 1]
    temperature = data[:, 2]
    x = data[:, 3] * microns_per_pixel
    x -= np.mean(x)
    y = data[:, 4] * microns_per_pixel
    y -= np.mean(y)
    data_length = len(x)
    x_image = x[0:data_length:1000]
    y_image = y[0:data_length:1000]
    t_image = t[0:data_length:1000]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twinx()
    ax2.plot(t, temperature, "b-")
    ax1.set_xlabel(r'Time [$\mathrm{s}$]')
    ax2.set_ylabel(r'Temperature [$^\circ$C]')
    if axis == "x":
        ax1.plot(t, x, "r-")
        line, = ax1.plot(t_image, x_image, ls = "", marker = "o")
        ax1.set_ylabel(r'X Position [$\mathrm{\mu m}$]')
    elif axis == "y":
        ax1.plot(t, y, "r-")
        line, = ax1.plot(t_image, y_image, ls = "", marker = "o")
        ax1.set_ylabel(r'Y Position [$\mathrm{\mu m}$]')
    else:
        print("data does not exist")

    def click(event):
        # if the mouse is clicked
        if line.contains(event)[0]:
            # find out the index within the array from the event
            ind, = line.contains(event)[1]["ind"]
            cv2.imshow('Original_Image_%s' % time.strftime("%M%S"), np.array(original_image[ind, :, :], dtype = np.uint8))
            cv2.imshow('Correlation_Image_%s' % time.strftime("%M%S"), np.array(corr_image[ind, :, :, :], dtype = np.uint8))
            k = cv2.waitKey(0)
            # wait for ESC key to exit
            if k == 27:
                cv2.destroyAllWindows()
            else:
                pass
        else:
            pass

    # add callback for mouse clicks
    fig1.canvas.mpl_connect('button_press_event', click)
    plt.show()

df.close()
