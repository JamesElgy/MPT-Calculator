import numpy as np
from scipy.special import jv
from matplotlib import pyplot as plt


def polarisation(omega, epsilon, sigma, mu, mu0, alpha):
    """
    Function to calculate the mpt for a sphere of radius alpha at radial frequency omega.
    Addapted from a Matlab function from Paul Ledger (exactsphererev5freqscan_nod.m).
    :param omega - Angular frequency rad/s
    :param epsilon - Permittivity:
    :param sigma - Conductivity S/m:
    :param mu - Permeability H/m:
    :param mu0 - Permeability of free space:
    :param alpha - Sphere radius m:
    :return eig - Single unique eigenvalue of the mpt tensor for a sphere of radius alpha:
    """

    k = np.sqrt((omega ** 2 * epsilon * sigma) + (mu * sigma * omega) * 1j)

    js_0_kr = np.sqrt(np.pi / (2 * k * alpha)) * jv(1 / 2, k * alpha)  # jv is the bessel function of the first kind.
    js_1_kr = np.sqrt(np.pi / (2 * k * alpha)) * jv(3 / 2, k * alpha)
    js_2_kr = np.sqrt(np.pi / (2 * k * alpha)) * jv(5 / 2, k * alpha)

    Ip12 = np.sqrt(2 / np.pi / (k * alpha)) * np.sinh(k * alpha)
    Im12 = np.sqrt(2 / np.pi / (k * alpha)) * np.cosh(k * alpha)

    numerator = ((2 * mu + mu0) * k * alpha * Im12 - (mu0 * (1 + (k * alpha) ** 2) + 2 * mu) * Ip12)
    denominator = ((mu - mu0) * k * alpha * Im12 + (mu0 * (1 + (k * alpha) ** 2) - mu) * Ip12)
    eig = 2 * np.pi * alpha ** 3 * numerator / denominator
    eig = np.conj(eig)

    return eig


def calc_sphere_mpt():
    # Setting Parameters:
    # free space permeability (in H/m)
    mu0 = 4 * np.pi * 1e-7
    # object relative permeability
    mur = 1.5
    # object permeability (in H/m)
    mu = mur * mu0
    # Angular Frequency (in rad/s)
    omega_start = 1e1
    omega_stop = 1e8
    n_samples = 81
    # Object conductivity (in S/m)
    sigma = 6e6
    # Sphere size (raduis) (in m)
    alpha = 0.01
    # Permittivity (eddy current approximation ~ neglected)
    epsilon = 0

    # Defining frequency sweep:
    omega = np.logspace(np.log10(omega_start), np.log10(omega_stop), n_samples)

    dataset = np.zeros((3, n_samples))
    for ind, o in enumerate(omega):
        mpt = polarisation(o, epsilon, sigma, mu, mu0, alpha)
        dataset[1, ind] = mpt.real
        dataset[2, ind] = mpt.imag
        dataset[0, ind] = o/2/np.pi

    plt.figure();
    plt.title('Real Component of M')
    plt.semilogx(dataset[0, :], dataset[1, :])
    plt.xlabel('Angular Frequency, (Hz)')
    plt.ylabel('Re(M)')
    # plt.show()

    plt.figure();
    plt.title('Imaginary Component of M')
    plt.semilogx(dataset[0, :], dataset[2, :])
    plt.xlabel('Angular Frequency, (Hz')
    plt.ylabel('Im(M)')
    plt.show()

    return dataset

def compare_to_external_dataset(filename):
    external_dataset = np.genfromtxt(filename, delimiter=',')[1:,:]
    internal_dataset = np.transpose(calc_sphere_mpt())

    plt.figure(); plt.title('real')
    ax1 = plt.subplot(211, xscale='log')
    ax2 = plt.subplot(212, xscale='log', sharex=ax1, sharey=ax1)
    ax1.plot(external_dataset[:,0], external_dataset[:,1], color='red', label='External dataset')
    ax1.legend()
    ax1.set_ylabel('Re(M)')

    ax2.plot(internal_dataset[:,0], internal_dataset[:,1], color='blue', label='Internal dataset')
    ax2.legend()
    ax2.set_ylabel('Re(M)')
    ax2.set_xlabel('Frequency, Hz')

    plt.figure(); plt.title('imag')
    ax3 = plt.subplot(211, xscale='log')
    ax4 = plt.subplot(212, xscale='log', sharex=ax3, sharey=ax3)
    ax3.plot(external_dataset[:,0], external_dataset[:,2], color='red', label='External dataset')
    ax3.legend()
    ax3.set_ylabel('Im(M)')

    ax4.plot(internal_dataset[:,0], internal_dataset[:,2], color='blue', label='Internal dataset')
    ax4.legend()
    ax4.set_ylabel('Im(M)')
    ax4.set_xlabel('Frequency, Hz')
    # plt.show()

    real_difference = external_dataset[:,1] - internal_dataset[:,1]
    imag_difference = external_dataset[:,2] - internal_dataset[:,2]

    plt.figure()
    ax5 = plt.subplot(211, xscale='log')
    ax6 = plt.subplot(212, xscale='log')
    ax5.plot(external_dataset[:,0], real_difference, label='$Re(External) - Re(Internal)$')
    ax5.legend()
    ax5.set_ylabel('$Re(External) - Re(Internal)$')
    ax6.plot(external_dataset[:,0], imag_difference, label='$Im(External) - Im(Internal)$')
    ax6.legend()
    ax6.set_ylabel('$Im(External) - Im(Internal)$')
    ax6.set_xlabel('Frequency, Hz')

    plt.show()

    print(f'real_difference std = {np.std(real_difference)}')
    print(f'imag_difference std = {np.std(imag_difference)}')

if __name__ == '__main__':
    calc_sphere_mpt()
    # compare_to_external_dataset('PLedger_sphere_mpt.txt')