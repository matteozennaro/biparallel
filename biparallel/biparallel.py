import numpy as np
from .core.core_functions import printmesh, compute_raw_bispectrum, compute_normalization, compute_effective_triangles

class BiParallel:
    def __init__(self, density_mesh, Lbox, nthreads=1, verbose=True):
        self.density_mesh = density_mesh
        self.Lbox = Lbox
        self.nthreads = nthreads
        self.verbose = verbose

    def print_mesh(self):
        nthreads = self.nthreads  # You can set this to the desired number of threads
        verbose = int(self.verbose)  # Set to True for detailed output
        printmesh(self.density_mesh, nthreads, verbose)

    def compute_raw_bispectrum(self, kbin_edges):
        # Compute all triangle configurations
        k1_min_bin = []
        k1_max_bin = []
        k2_min_bin = []
        k2_max_bin = []
        k3_min_bin = []
        k3_max_bin = []
        kbin_centers = 0.5 * (kbin_edges[:-1] + kbin_edges[1:])
        for i in range(len(kbin_edges) - 1):
            for j in range(i, len(kbin_edges) - 1):
                for k in range(j, len(kbin_edges) - 1):
                    if kbin_centers[k] <= kbin_centers[i] + kbin_centers[j]:
                        k1_min_bin.append(kbin_edges[i])
                        k1_max_bin.append(kbin_edges[i + 1])
                        k2_min_bin.append(kbin_edges[j])
                        k2_max_bin.append(kbin_edges[j + 1])
                        k3_min_bin.append(kbin_edges[k])
                        k3_max_bin.append(kbin_edges[k + 1])
        k1_min_bin = np.array(k1_min_bin, dtype=np.float32)
        k1_max_bin = np.array(k1_max_bin, dtype=np.float32)
        k2_min_bin = np.array(k2_min_bin, dtype=np.float32)
        k2_max_bin = np.array(k2_max_bin, dtype=np.float32)
        k3_min_bin = np.array(k3_min_bin, dtype=np.float32)
        k3_max_bin = np.array(k3_max_bin, dtype=np.float32)
        # Call the core function to compute the raw bispectrum
        bispectrum = compute_raw_bispectrum(self.density_mesh, self.Lbox, k1_min_bin, k1_max_bin, k2_min_bin, k2_max_bin, k3_min_bin, k3_max_bin, self.nthreads, int(self.verbose))
        return bispectrum

    def compute_normalization(self, kbin_edges):
        # Similar to compute_raw_bispectrum but calls a different core function for normalization
        k1_min_bin = []
        k1_max_bin = []
        k2_min_bin = []
        k2_max_bin = []
        k3_min_bin = []
        k3_max_bin = []
        kbin_centers = 0.5 * (kbin_edges[:-1] + kbin_edges[1:])
        for i in range(len(kbin_edges) - 1):
            for j in range(i, len(kbin_edges) - 1):
                for k in range(j, len(kbin_edges) - 1):
                    if kbin_centers[k] <= kbin_centers[i] + kbin_centers[j]:
                        k1_min_bin.append(kbin_edges[i])
                        k1_max_bin.append(kbin_edges[i + 1])
                        k2_min_bin.append(kbin_edges[j])
                        k2_max_bin.append(kbin_edges[j + 1])
                        k3_min_bin.append(kbin_edges[k])
                        k3_max_bin.append(kbin_edges[k + 1])
        k1_min_bin = np.array(k1_min_bin, dtype=np.float32)
        k1_max_bin = np.array(k1_max_bin, dtype=np.float32)
        k2_min_bin = np.array(k2_min_bin, dtype=np.float32)
        k2_max_bin = np.array(k2_max_bin, dtype=np.float32)
        k3_min_bin = np.array(k3_min_bin, dtype=np.float32)
        k3_max_bin = np.array(k3_max_bin, dtype=np.float32)
        # Call the core function to compute the normalization
        ngrid = self.density_mesh.shape[0]
        normalization = compute_normalization(ngrid, self.Lbox, k1_min_bin, k1_max_bin, k2_min_bin, k2_max_bin, k3_min_bin, k3_max_bin, self.nthreads, int(self.verbose))
        return normalization

    def compute_effective_triangles(self, kbin_edges):
        k1_min_bin = []
        k1_max_bin = []
        k2_min_bin = []
        k2_max_bin = []
        k3_min_bin = []
        k3_max_bin = []
        kbin_centers = 0.5 * (kbin_edges[:-1] + kbin_edges[1:])
        for i in range(len(kbin_edges) - 1):
            for j in range(i, len(kbin_edges) - 1):
                for k in range(j, len(kbin_edges) - 1):
                    if kbin_centers[k] <= kbin_centers[i] + kbin_centers[j]:
                        k1_min_bin.append(kbin_edges[i])
                        k1_max_bin.append(kbin_edges[i + 1])
                        k2_min_bin.append(kbin_edges[j])
                        k2_max_bin.append(kbin_edges[j + 1])
                        k3_min_bin.append(kbin_edges[k])
                        k3_max_bin.append(kbin_edges[k + 1])
        ngrid = self.density_mesh.shape[0]
        effective_triangles = compute_effective_triangles(ngrid, self.Lbox, k1_min_bin, k1_max_bin, k2_min_bin, k2_max_bin, k3_min_bin, k3_max_bin, self.nthreads, int(self.verbose))
        return effective_triangles

    def compute_bispectrum(self, kbin_edges):
        raw_bispectrum = self.compute_raw_bispectrum(kbin_edges)
        normalization = self.compute_normalization(kbin_edges)
        bispectrum = raw_bispectrum / normalization * self.Lbox**6 / self.density_mesh.size**3
        return bispectrum