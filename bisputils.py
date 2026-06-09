import numpy as np
import bacco

class BaccoBispectrum:
    def __init__(self, sim):
        self.sim = sim
        self.Cosmology = sim.Cosmology

    def compute_bispectrum_BiG(self, ngrid, kmin, kmax, dk, binning_mode='linear', compensated=True):
        import os
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        import jax
        print(jax.devices())
        from BiG import bispectrumExtractor as BiG

        assert binning_mode in ['linear', 'log', 'hybrid']

        density_grid = self.compute_density_mesh(ngrid, compensated=compensated)

        L_folded = self.sim.header['BoxSize']
        Nmesh = ngrid
        verbose = True
        mode = 'all'
        filetype = 'direct'
        k_f = 2.0 * np.pi / L_folded

        if binning_mode=='log':
            dlogk = dk
            kbins_log = np.arange(np.log10(kmin), np.log10(kmax), dlogk)
            kbins_log = 10**kbins_log
            # kbins = np.unique((kbins_log/kmin).astype(int))*kmin + kmin
            kbins = kbins_log
        elif binning_mode=='hybrid':
            dlink, dlogk = dk[0], dk[1]
            kbins_lin = np.arange(kmin[0], kmax[0], dlink)
            kbins_log = np.arange(np.log10(kmin[1]), np.log10(kmax[1]), dlogk)
            kbins_log = 10**kbins_log
            kbins = np.concatenate([kbins_lin, kbins_log])
        else:
            nstep = int(np.floor((kmax-kmin)/dk))
            kbins = np.array([kmin + i*dk for i in range(nstep+1)])

        print(f'N bins: {len(kbins)-1}')
        print(f'kbins: {kbins}')

        kbins_lower=kbins[:-1]
        kbins_upper=kbins[1:]
        kbins_mid=0.5*(kbins_lower+kbins_upper)
        kbinedges=[kbins_lower, kbins_upper, kbins_mid]

        self.kbins_lower = kbins_lower
        self.kbins_upper = kbins_upper
        self.kbins_mid = kbins_mid
        self.kbinedges = kbinedges

        Xtract = BiG.bispectrumExtractor(L_folded, Nmesh, kbinedges, verbose)
        norm = np.array(Xtract.calculateBispectrumNormalization_slow(mode=mode))
        prefactor = Xtract.prefactor
        eff_triangles = np.array(Xtract.calculateEffectiveTriangle_slow(mode=mode)).T / norm
        bkkk = np.array(Xtract.calculateBispectrum_slow(density_grid, mode=mode, filetype=filetype)) / norm * prefactor

        pkprefactor = L_folded**3 / Nmesh**6
        pknorm = np.array(Xtract.calculatePowerspectrumNormalization())
        pknorm /= pkprefactor
        powerspec = np.array(Xtract.calculatePowerspectrum(density_grid, filetype=filetype)) / pknorm

        return {'effective_triangles': eff_triangles, 'bispectrum': bkkk, 'powerspectrum': powerspec, 'kbins': kbins, 'kbins_mid': kbins_mid, 'norm': norm, 'prefactor': prefactor}


    def compute_crosspower_BiG(self, mesh1, mesh2, ngrid, kmin, kmax, dk, log_binning=False):
        import os
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        import jax
        print(jax.devices())
        from BiG import bispectrumExtractor as BiG

        L_folded = self.sim.header['BoxSize']
        Nmesh = ngrid
        verbose = True
        mode = 'all'
        filetype = 'direct'
        k_f = 2.0 * np.pi / L_folded

        if log_binning:
            dlogk = dk
            kbins_log = np.arange(np.log10(kmin), np.log10(kmax), dlogk)
            kbins_log = 10**kbins_log
             # kbins = np.unique((kbins_log/k_f).astype(int))*k_f + k_f
            kbins = np.unique((kbins_log/kmin).astype(int))*kmin + kmin
        else:
            nstep = int(np.floor((kmax-kmin)/dk))
            kbins = np.array([kmin + i*dk for i in range(nstep+1)])

        print(f'N bins: {len(kbins)-1}')
        print(f'kbins: {kbins}')

        kbins_lower=kbins[:-1]
        kbins_upper=kbins[1:]
        kbins_mid=0.5*(kbins_lower+kbins_upper)
        kbinedges=[kbins_lower, kbins_upper, kbins_mid]

        Xtract = BiG.bispectrumExtractor(L_folded, Nmesh, kbinedges, verbose)

        pkprefactor = L_folded**3 / Nmesh**6
        pknorm = np.array(Xtract.calculatePowerspectrumNormalization())
        pknorm /= pkprefactor
        powerspec = np.array(Xtract.calculateCrossPowerspectrum(mesh1, mesh2, filetype=filetype)) / pknorm

        return {'powerspectrum': powerspec, 'kbins': kbins, 'kbins_mid': kbins_mid}


    def compute_density_mesh(self, ngrid, compensated=True):
        density_grid = bacco.statistics.compute_mesh(ngrid=ngrid, box=self.sim.header['BoxSize'], pos=self.sim.sdm['pos'], vel=None,
                                                     interlacing=True, deposit_method='tsc', folds=1, compensated=compensated,
                                                     zspace=False, cosmology=self.Cosmology, nthreads=None, twod=False)
        density_grid /= np.mean(density_grid)
        density_grid -= 1
        return density_grid

    def compute_bihalofit(self, bispectrum_results):
        k123 = bispectrum_results['effective_triangles']
        k1 = k123[2]
        k2 = k123[1]
        k3 = k123[0]
        bihalofit_bkkk = self.sim.Cosmology.compute_bihalofit(k1, k2, k3)
        return bihalofit_bkkk





class BaccoBispectrumEquilateral(BaccoBispectrum):
    def compute_bispectrum_BiG(self, ngrid, kmin, kmax, dk, log_binning=False, compensated=True):
        import os
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        import jax
        print(jax.devices())
        from BiG import bispectrumExtractor as BiG

        density_grid = self.compute_density_mesh(ngrid, compensated=compensated)

        L_folded = self.sim.header['BoxSize']
        Nmesh = ngrid
        verbose = True
        mode = 'equilateral'
        filetype = 'direct'
        k_f = 2.0 * np.pi / L_folded

        if log_binning:
            dlogk = dk
            kbins_log = np.arange(np.log10(kmin), np.log10(kmax), dlogk)
            kbins_log = 10**kbins_log
            # kbins = np.unique((kbins_log/k_f).astype(int))*k_f + k_f
            kbins = np.unique((kbins_log/kmin).astype(int))*kmin + kmin
        else:
            nstep = int(np.floor((kmax-kmin)/dk))
            kbins = np.array([kmin + i*dk for i in range(nstep+1)])

        print(f'N bins: {len(kbins)-1}')
        print(f'kbins: {kbins}')

        kbins_lower=kbins[:-1]
        kbins_upper=kbins[1:]
        kbins_mid=0.5*(kbins_lower+kbins_upper)
        kbinedges=[kbins_lower, kbins_upper, kbins_mid]

        Xtract = BiG.bispectrumExtractor(L_folded, Nmesh, kbinedges, verbose)
        norm = np.array(Xtract.calculateBispectrumNormalization_slow(mode=mode))
        prefactor = Xtract.prefactor
        eff_triangles = np.array(Xtract.calculateEffectiveTriangle_slow(mode=mode)).T / norm
        bkkk = np.array(Xtract.calculateBispectrum_slow(density_grid, mode=mode, filetype=filetype)) / norm * prefactor

        pkprefactor = L_folded**3 / Nmesh**6
        pknorm = np.array(Xtract.calculatePowerspectrumNormalization())
        pknorm /= pkprefactor
        powerspec = np.array(Xtract.calculatePowerspectrum(density_grid, filetype=filetype)) / pknorm

        return {'effective_triangles': eff_triangles, 'bispectrum': bkkk, 'powerspectrum': powerspec, 'kbins': kbins, 'kbins_mid': kbins_mid}