import os
import numpy as np
from .core.core_functions import printmesh, compute_raw_bispectrum, compute_normalization, compute_effective_triangles, set_wisdom_path

class BiParallel:
    def __init__(self, density_mesh, Lbox, nthreads=None, verbose=True, wisdom_path=None):
        self.density_mesh = density_mesh
        self.Lbox = Lbox
        self.nthreads = nthreads if nthreads is not None else os.cpu_count()
        self.verbose = verbose
        self.wisdom_path = self._resolve_wisdom_path(wisdom_path)
        self.wisdom_exists = os.path.exists(self.wisdom_path)
        set_wisdom_path(self.wisdom_path)

        if self.verbose:
            state = "found" if self.wisdom_exists else "not found"
            print(f"FFTW wisdom {state}: {self.wisdom_path}")

    def _resolve_wisdom_path(self, wisdom_path):
        if wisdom_path is not None:
            return os.path.abspath(os.path.expanduser(wisdom_path))

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "biparallel")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "fftw_wisdom.dat")

    def print_mesh(self):
        nthreads = self.nthreads  # You can set this to the desired number of threads
        verbose = int(self.verbose)  # Set to True for detailed output
        printmesh(self.density_mesh, nthreads, verbose)

    def _compute_triangle_bins(self, kbin_edges):
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
        return (np.array(k1_min_bin, dtype=np.float32), np.array(k1_max_bin, dtype=np.float32),
                np.array(k2_min_bin, dtype=np.float32), np.array(k2_max_bin, dtype=np.float32),
                np.array(k3_min_bin, dtype=np.float32), np.array(k3_max_bin, dtype=np.float32))

    def compute_raw_bispectrum(self, kbin_edges):
        # Compute all triangle configurations
        k1_min_bin, k1_max_bin, k2_min_bin, k2_max_bin, k3_min_bin, k3_max_bin = self._compute_triangle_bins(kbin_edges)
        # Call the core function to compute the raw bispectrum
        self.raw_bispectrum = compute_raw_bispectrum(self.density_mesh, self.Lbox, k1_min_bin, k1_max_bin, k2_min_bin, k2_max_bin, k3_min_bin, k3_max_bin, self.nthreads, int(self.verbose))
        return self.raw_bispectrum

    def compute_normalization(self, kbin_edges):
        # Similar to compute_raw_bispectrum but calls a different core function for normalization
        k1_min_bin, k1_max_bin, k2_min_bin, k2_max_bin, k3_min_bin, k3_max_bin = self._compute_triangle_bins(kbin_edges)
        # Call the core function to compute the normalization
        ngrid = self.density_mesh.shape[0]
        self.normalization = compute_normalization(ngrid, self.Lbox, k1_min_bin, k1_max_bin, k2_min_bin, k2_max_bin, k3_min_bin, k3_max_bin, self.nthreads, int(self.verbose))
        return self.normalization

    def compute_effective_triangles(self, kbin_edges):
        # Similar to compute_raw_bispectrum but calls a different core function for effective triangles
        k1_min_bin, k1_max_bin, k2_min_bin, k2_max_bin, k3_min_bin, k3_max_bin = self._compute_triangle_bins(kbin_edges)
        # Call the core function to compute the effective triangles
        ngrid = self.density_mesh.shape[0]
        effective_triangles = compute_effective_triangles(ngrid, self.Lbox, k1_min_bin, k1_max_bin, k2_min_bin, k2_max_bin, k3_min_bin, k3_max_bin, self.nthreads, int(self.verbose))
        normalization = self.compute_normalization(kbin_edges) if not hasattr(self, 'normalization') else self.normalization
        self.effective_triangles = effective_triangles / normalization
        return self.effective_triangles

    def compute_bispectrum(self, kbin_edges):
        raw_bispectrum = self.compute_raw_bispectrum(kbin_edges) if not hasattr(self, 'raw_bispectrum') else self.raw_bispectrum
        normalization = self.compute_normalization(kbin_edges) if not hasattr(self, 'normalization') else self.normalization
        bispectrum = raw_bispectrum / normalization * self.Lbox**6 / self.density_mesh.size**3
        effective_triangles = self.compute_effective_triangles(kbin_edges) if not hasattr(self, 'effective_triangles') else self.effective_triangles
        return {'bispectrum': bispectrum, 'effective_triangles': effective_triangles}