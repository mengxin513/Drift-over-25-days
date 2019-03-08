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

def allan_variance(data, dt, npoints):
    """Calculate the Allan variance of 1D data, as explained in the paper."""
    assert data.ndim == 1
    datapoints = data.shape[0]
    blocksizes = np.round(np.exp(np.linspace(0, 1, npoints) * np.log(datapoints / 5))).astype(np.int) #logarithmically seperates
    allan_var = np.zeros((npoints, 2))

    tau = blocksizes * dt
    
    for i, b in enumerate(blocksizes):
        data_blocked = data[0:(datapoints // b) * b].reshape((-1, b))
        xis = np.mean(data_blocked, axis = 1)
        allan_var[i, 0] = tau[i]
        allan_var[i, 1] = 0.5 * np.sqrt(np.mean(np.diff(xis) ** 2))
    return allan_var

if __name__ == "__main__":
    with PdfPages('Allan_deviation.pdf') as pdf:

        print ('Loading data...') #indication of the programme running

        microns_per_pixel = 2.74
        N_frames = 500
        #need to be consistant between drift.py and drift_plot.py

        df = h5py.File('drift.hdf5', mode = 'r')
        group = list(df.values())[-1]
        N_points = len(group) - 2
        data = np.zeros([N_frames * N_points, 3])
        for i in range(N_points):
            dset = group['data%05d' % i]
            for j in range(N_frames):
                data[N_frames * i + j, 0] = dset[j, 0]
                data[N_frames * i + j, 1] = dset[j, 3]
                data[N_frames * i + j, 2] = dset[j, 4]
            printProgressBar(i, N_points)
        print('')

        matplotlib.rcParams.update({'font.size': 12})

        filtered_data = data#[10000:10050, :]

        print('Number of data points: {}'.format(len(filtered_data)))
        print(filtered_data)

        dt = np.mean(np.diff(filtered_data[:, 0]))

        allan_x = allan_variance(filtered_data[:, 1] * microns_per_pixel, dt, 200)
        allan_y = allan_variance(filtered_data[:, 2] * microns_per_pixel, dt, 200)

        fig, ax = plt.subplots(1, 1)

        line1, = ax.loglog(allan_x[:, 0], allan_x[:, 1], 'ro-')
        line2, = ax.loglog(allan_y[:, 0], allan_y[:, 1], 'bo-')

        ax.legend((line1, line2), ('X', 'Y'))

        ax.set_xlabel(r'Time [$\mathrm{s}$]')
        ax.set_ylabel(r'Allan Deviation [$\mathrm{\mu m}$]')

        plt.tight_layout()

        pdf.savefig(fig)
        plt.show()
        plt.close(fig)

        for a, b in enumerate(['X', 'Y']):
            fig2, ax2 = plt.subplots(1, 1)
            hist, bins = np.histogram(data[:, a]%1, 500)

            ax2.plot(bins[1:], hist)

            ax2.set_xlabel('Pixel remainder along ' + b + r' [$\mathrm{px}$]')
            ax2.set_ylabel('Counts')

            plt.tight_layout()

            pdf.savefig(fig2)
            plt.close(fig2)

    df.close()
