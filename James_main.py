import os
import sys
import numpy as np
import main as MPT_main
from matplotlib import pyplot as plt

sys.path.insert(0, 'JamesAddons')
import exact_sphere


def plot_sphere_eigenvalue_comparison(FEM_dataset, exact_dataset):

    plt.figure(); plt.title('real')
    ax1 = plt.subplot(211, xscale='log')
    ax2 = plt.subplot(212, xscale='log', sharex=ax1, sharey=ax1)
    ax1.plot(FEM_dataset[:, 0], FEM_dataset[:, 1], color='red', label='External dataset')
    ax1.legend()
    ax1.set_ylabel('Re(M)')

    ax2.plot(exact_dataset[:, 0], exact_dataset[:, 1], color='blue', label='Internal dataset')
    ax2.legend()
    ax2.set_ylabel('Re(M)')
    ax2.set_xlabel('Frequency, Hz')

    plt.figure(); plt.title('imag')
    ax3 = plt.subplot(211, xscale='log')
    ax4 = plt.subplot(212, xscale='log', sharex=ax3, sharey=ax3)
    ax3.plot(FEM_dataset[:, 0], FEM_dataset[:, 2], color='red', label='External dataset')
    ax3.legend()
    ax3.set_ylabel('Im(M)')

    ax4.plot(exact_dataset[:,0], exact_dataset[:, 2], color='blue', label='Internal dataset')
    ax4.legend()
    ax4.set_ylabel('Im(M)')
    ax4.set_xlabel('Frequency, Hz')
    plt.show()

def generate_FEM_exact_sphere_datasets():
    TensorArray, EigenValues, N0, elements, array = MPT_main.main()
    dataset = exact_sphere.calc_sphere_mpt()
    dataset = dataset.transpose()

    dataset_MPT_calculator = np.zeros((len(EigenValues[:, 0]), 3))
    dataset_MPT_calculator[:, 0] = dataset[:,0]
    dataset_MPT_calculator[:, 1] = EigenValues[:, 0].real
    dataset_MPT_calculator[:, 2] = EigenValues[:, 0].imag

    return dataset_MPT_calculator, dataset

if __name__ == '__main__':
    dataset_MPT_calculator, dataset = generate_FEM_exact_sphere_datasets()
    plot_sphere_eigenvalue_comparison(dataset_MPT_calculator, dataset)