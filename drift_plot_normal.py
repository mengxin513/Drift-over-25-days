from matplotlib.backends.backend_pdf import PdfPages
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def printProgressBar(iteration, total, length = 10):
    percent = 100.0 * iteration / total
    filledLength = int(length * iteration // total)
    bar = '*' * filledLength + '-' * (length - filledLength)
    print('Progress: |%s| %d%% Completed' % (bar, percent), end = '\r')
    if iteration == total: 
        print()

def graph(a, b, c, xlabel, ylabel1, ylabel2, group):
    fig, ax = plt.subplots(1, 1)

    ax.plot(a, b, 'r-')
    ax2 = ax.twinx()
    ax2.plot(a, c, 'b-')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel1)
    ax2.set_ylabel(ylabel2)

    fig.suptitle(group.name[1:])

    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

if __name__ == "__main__":
    with PdfPages('drift.pdf') as pdf:

        print ('Loading data...')

        N_frames = 1000
        #need to be consistant between drift.py and drift_plot.py

        df = h5py.File('drift.hdf5', mode = 'r')
        group = list(df.values())[-1]
        N_points = len(group) - 2
        data = np.zeros([N_frames * N_points, 5])
        for i in range(N_points):
            dset = group['data%05d' % i]
            for j in range(N_frames):
                data[N_frames * i + j, :] = dset[j, :]
            printProgressBar(i, N_points)
        print('')

        print('Number of data points: {}'.format(len(data)))
        print(data)

        matplotlib.rcParams.update({'font.size': 12})

        microns_per_pixel = 2.74

        time = data[:, 0] / (60 * 60 * 24)
        humidity = data[:, 1]
        temperature = data[:, 2]
        x = data[:, 3] * microns_per_pixel
        x -= np.mean(x)
        y = data[:, 4] * microns_per_pixel
        y -= np.mean(y)

        #graph(time, x, temperature, r'Time [$\mathrm{s}$]', r'X Position [$\mathrm{\mu m}$]', r'Temperature [$^\circ$C]', group)
        #graph(time, y, temperature, r'Time [$\mathrm{s}$]', r'Y Position [$\mathrm{\mu m}$]', r'Temperature [$^\circ$C]', group)

        fig, ax = plt.subplots(1, 1)

        line1, = ax.plot(time, x, 'r-')
        line2, = ax.plot(time, y, 'b-')
        ax2 = ax.twinx()
        line3, = ax2.plot(time, temperature, 'k-', alpha = 0.4)

        ax.legend((line1, line2, line3), ('X', 'Y', 'Temperature'))

        ax.set_xlabel(r'Time [$\mathrm{Days}$]')
        ax.set_ylabel(r'Stage Position [$\mathrm{\mu m}$]')
        ax2.set_ylabel(r'Temperature [$^\circ$C]')

        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

    df.close()
