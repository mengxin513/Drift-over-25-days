from matplotlib.backends.backend_pdf import PdfPages
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with PdfPages("tracking_analysis.pdf") as pdf:

        print ("Loading data...")

        df = h5py.File("tracking_analysis.hdf5", mode = "r")
        group = list(df.values())[-1]
        dset = group["data00000"]
        n = len(dset)
        data = np.zeros([n, 3])
        for i in range(n):
            data[i, :] = dset[i, :]

        matplotlib.rcParams.update({'font.size': 12})

        microns_per_pixel = 2.74

        max_random_noise = data[:, 0]
        change_in_x = (data[:, 1] - data[0, 1]) * microns_per_pixel
        change_in_y = (data[:, 2] - data[0, 2]) * microns_per_pixel

        fig, ax = plt.subplots(1, 1)

        line1, = ax.plot(max_random_noise, change_in_x, "ro-")
        line2, = ax.plot(max_random_noise, change_in_y, "bo-")

        ax.legend((line1, line2), ('X', 'Y'))

        ax.set_xlabel(r'Max added random noise')
        ax.set_ylabel(r'Change in position [$\mathrm{\mu m}$]')

        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

    df.close()
