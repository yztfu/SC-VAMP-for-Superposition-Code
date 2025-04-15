import logging
import math
from scipy.fft import dct, idct
import numpy as np


class DenoiserBayesFunc:
    def __init__(self, x_prior_type:str='bgauss', pa_type:str=None, **kwargs) -> None:
        if x_prior_type == 'bgauss':
            self.denoiser_bayes = self.denoiser_bayes_bgauss
        elif x_prior_type == 'src':
            if not pa_type:
                self.denoiser_bayes = self.denoiser_bayes_src
            else:
                self.denoiser_bayes = self.denoiser_bayes_src_pa
        else:
            raise ValueError("Type of x_prior does not exist!")

    def denoiser_bayes_bgauss(self, beta_pri=None, var_pri=None, rho=None, **kwargs):
        exps1 = np.exp(- beta_pri**2 / (2*(var_pri+1/rho)))
        exps2 = np.exp(- beta_pri**2 / (2*var_pri))
        ratio1 = np.sqrt(var_pri / (var_pri+1/rho))

        beta_post_top = ratio1 * beta_pri * exps1 / (var_pri+1/rho)
        beta_post_bot = (1 - rho) * exps2 + rho * ratio1 * exps1
        beta_post = beta_post_top / beta_post_bot

        var_post_top = rho * (1 - rho) * ratio1 * exps1 * exps2 * (var_pri/rho*(var_pri+1/rho) + (beta_pri/rho)**2) / (1/rho+var_pri)**2 \
            + rho * exps1**2 * var_pri**2 / (var_pri+1/rho)**2
        var_post_bot = beta_post_bot ** 2
        var_post = var_post_top / var_post_bot
        var_post_mean = np.mean(var_post)
        return beta_post, var_post_mean

    def denoiser_bayes_src(self, beta_pri=None, var_pri=None, N_all=None, B_src=None, eps=None, **kwargs):
        # assert N_src % B_src == 0, "N_src must be a multiple of B_src."
        L_src = N_all // B_src
        rt_n_Pl = np.sqrt(B_src) * np.ones((N_all, 1))
        u = beta_pri * rt_n_Pl / (var_pri + eps)
        max_u = u.reshape(L_src, B_src).max(axis=1).repeat(B_src).reshape(-1, 1)
        exps = np.exp(u - max_u)
        sums = exps.reshape(L_src, B_src).sum(axis=1).repeat(B_src).reshape(-1, 1)
        beta_post = (rt_n_Pl * exps / sums).reshape(-1, 1)
        var_post = (rt_n_Pl ** 2 * (exps / sums) * (1 - exps / sums)).reshape(-1, 1)
        var_post_mean = np.mean(var_post)
        return beta_post, var_post_mean

    def denoiser_bayes_src_pa(self, beta_pri=None, var_pri=None, N_all=None, B_src=None, eps=None, Pl=None, **kwargs):
        # assert N_src % B_src == 0, "N_src must be a multiple of B_src."
        L_src = N_all // B_src
        rt_n_Pl = np.sqrt(N_all * Pl).repeat(B_src).reshape(-1, 1)
        u = beta_pri * rt_n_Pl / (var_pri + eps)
        max_u = u.reshape(L_src, B_src).max(axis=1).repeat(B_src).reshape(-1, 1)
        exps = np.exp(u - max_u)
        sums = exps.reshape(L_src, B_src).sum(axis=1).repeat(B_src).reshape(-1, 1)
        beta_post = (rt_n_Pl * exps / sums).reshape(-1, 1)
        var_post = (rt_n_Pl ** 2 * (exps / sums) * (1 - exps / sums)).reshape(-1, 1)
        var_post_mean = np.mean(var_post)
        return beta_post, var_post_mean

    def __call__(self, **kwargs):
        return self.denoiser_bayes(**kwargs)


class LMMSEEstiFunc:
    def __init__(self, lmmse_type='residual', **kwargs) -> None:
        if lmmse_type == 'residual':
            self.lmmse_esti = self.lmmse_esti_residual
        elif lmmse_type == 'matrix':
            self.lmmse_esti = self.lmmse_esti_matrix
        elif lmmse_type == 'vector':
            self.lmmse_esti = self.lmmse_esti_vector
        else:
            raise ValueError("Type of lmmse_type does not exist!")

    def lmmse_esti_residual(self, r1=None, y=None, gamma1=None, snr=None, A_mat=None, se=False, **kwargs):
        if not se:
            Mr, Nr = A_mat.shape
            F_mat = np.linalg.inv(gamma1 / snr * np.identity(Mr) + A_mat @ A_mat.T) @ A_mat
            x1 = r1 + F_mat.T @ (y - A_mat @ r1)
            alpha1 = 1 / Nr * np.trace(np.identity(Nr) - F_mat.T @ A_mat)
            return x1, alpha1
        else:
            Mr, Nr = A_mat.shape
            F_mat = np.linalg.inv(gamma1 / snr * np.identity(Mr) + A_mat @ A_mat.T) @ A_mat
            alpha1 = 1 / Nr * np.trace(np.identity(Nr) - F_mat.T @ A_mat)
            return alpha1

    def lmmse_esti_matrix(self, r1=None, y=None, gamma1=None, snr=None, A_mat=None, se=False, **kwargs):
        if not se:
            _, Nr = A_mat.shape
            F_mat = np.linalg.inv(snr * A_mat.T @ A_mat + gamma1 * np.identity(Nr))
            x1 = F_mat @ (snr * A_mat.T @ y + gamma1 * r1)
            alpha1 = gamma1 / Nr * np.trace(F_mat)
            return x1, alpha1
        else:
            _, Nr = A_mat.shape
            F_mat = np.linalg.inv(snr * A_mat.T @ A_mat + gamma1 * np.identity(Nr))
            alpha1 = gamma1 / Nr * np.trace(F_mat)
            return alpha1

    def lmmse_esti_vector(self, r1=None, y=None, gamma1=None, snr=None, A_mat=None, se=False, **kwargs):
        if not se:
            Mr, Nr = A_mat.shape
            x1 = r1 + snr / (snr + gamma1) * A_mat.T @ (y - A_mat @ r1)
            alpha1 = 1 - (Mr / Nr) * (snr / (snr + gamma1))
            return x1, alpha1
        else:
            Mr, Nr = A_mat.shape
            alpha1 = 1 - (Mr / Nr) * (snr / (snr + gamma1))
            return alpha1

    def __call__(self, **kwargs):
        return self.lmmse_esti(**kwargs)


class SignalSettings:
    def __init__(self, algo_ori:bool=False, algo_pa:bool=False, algo_sc:bool=False,
                 x_prior_type:str="bgauss", pa_type:str=None, N_all:int=1, sigma2:float=1, rho:float=1,
                 B_src:int=1, N:int=1, N_c:list[int]=None, W_cal:list=None, sfR:int=1, sfC:int=1, **kwargs) -> None:
        self.algo_ori, self.algo_pa, self.algo_sc, self.x_prior_type, self.pa_type = algo_ori, algo_pa, algo_sc, x_prior_type, pa_type
        # x_all: prior signal
        if x_prior_type == 'bgauss':
            x_all = self.x_prior_bgauss(N_all=N_all, rho=rho)
            if algo_ori:
                self.x_all_ori = x_all
                self.x_prior_generate_ori = self.x_prior_bgauss
            if algo_sc:
                self.x_all_sc = x_all
                self.x_prior_generate_sc = self.x_prior_bgauss
        elif x_prior_type == 'src':
            if not pa_type:
                x_all = self.x_prior_src(N_all=N_all, B_src=B_src)
                if algo_ori:
                    self.x_all_ori = x_all
                    self.x_prior_generate_ori = self.x_prior_src
                if algo_sc:
                    self.x_all_sc = x_all
                    self.x_prior_generate_sc = self.x_prior_src
            else:
                assert algo_pa is True, "The algo_pa must be True!"
                self.Pl, self.x_all_pa, self.x_all_ave = self.x_prior_src_pa(N_all=N_all, B_src=B_src, pa_type=pa_type, sigma2=sigma2, ifall=True)
                if algo_ori:
                    self.x_all_ori = self.x_all_ave
                    self.x_prior_generate_ori = self.x_prior_src
                if algo_sc:
                    self.x_all_sc = self.x_all_ave
                    self.x_prior_generate_sc = self.x_prior_src
        else:
            raise ValueError("Type of signal does not exist!")

        if algo_sc:
            self.x_vec, self.x_ra_vec = self.x_vec_generate(self.x_all_sc, N, N_c, W_cal, sfR, sfC)

    def x_prior_bgauss(self, N_all:int=1, rho:float=1, **kwargs):
        x_all = np.random.binomial(1, rho, (N_all, 1)) * np.random.normal(0, np.sqrt(1/rho), (N_all, 1))
        return x_all

    def x_prior_src(self, N_all:int=1, B_src:int=1, **kwargs):
        L_src = N_all // B_src
        tx_message = np.random.randint(0, B_src, L_src)
        x_all = np.zeros((N_all, 1))
        for l in range(L_src):
            x_all[l * B_src + tx_message[l], 0] = np.sqrt(B_src)
        return x_all

    def x_prior_src_pa(self, N_all:int=1, B_src:int=1, pa_type:str='exp', sigma2:float=1, ifall=False, **kwargs):
        L_src = N_all // B_src
        if pa_type == 'exp':
            C = 0.5 * np.log2(1 + 1 / sigma2)
            Pl = 2 ** (-2 * C * np.arange(L_src) / L_src)
            Pl = Pl / np.sum(Pl)
        elif pa_type == 'average':
            Pl = np.repeat(1 / L_src, L_src)
        else:
            raise ValueError("Type of power allocation does not exist!")

        tx_message = np.random.randint(0, B_src, L_src)
        x_all_pa = np.zeros((N_all, 1))
        x_all_ave = np.zeros((N_all, 1))
        for l in range(L_src):
            x_all_pa[l * B_src + tx_message[l], 0] = np.sqrt(N_all * Pl[l])
            x_all_ave[l * B_src + tx_message[l], 0] = np.sqrt(B_src)

        if ifall:
            return Pl, x_all_pa, x_all_ave
        else:
            return x_all_pa

    def x_vec_generate(self, x_all, N, N_c, W_cal, sfR, sfC):
        x_vec = [np.zeros((N, 1)) for _ in range(sfC)]
        x_ra_vec = [np.zeros((N_c[sfr], 1)) for sfr in range(sfR)]
        for sfc in range(sfC):
            x_vec[sfc] = x_all[N*sfc:N*(sfc+1)]
        for sfr in range(sfR):
            x_ra_vec[sfr] = np.concatenate([x_vec[w] for w in W_cal[sfr]], axis=0)
        return x_vec, x_ra_vec


class NoiseSettings:
    def __init__(self, noise_type:str="awgnc", M_all:int=1, sigma2:float=1, **kwargs) -> None:
        # n_all_fc: noise function, y = n_all_fc(z)
        assert noise_type == 'awgnc', "The noise_type must be awgnc!"
        self.noise_all_fc = self.noise_fc_awgnc(M_all, sigma2)

    def noise_fc_awgnc(self, M_all:int=1, sigma2:float=1):
        n_all = np.random.normal(0, np.sqrt(sigma2), (M_all, 1))
        def noise_all_fc(z_all):
            return z_all + n_all
        return noise_all_fc


class CodingSettings:
    def __init__(self, signal_inst:SignalSettings, noise_inst:NoiseSettings, mat_type:str="gauss",
                 M_all:int=1, N_all:int=1, M:int=1, N_c:list[int]=None, sfR:int=1, **kwargs) -> None:
        self.algo_ori, self.algo_pa, self.algo_sc = signal_inst.algo_ori, signal_inst.algo_pa, signal_inst.algo_sc
        # A_all/A_mat: sensing matrix of sc; A_ori: sensing matrix of original oamp
        if mat_type == 'gauss':
            if self.algo_ori or self.algo_pa:
                self.A_ori = self.mat_gauss_ori(M_all, N_all)
            if self.algo_sc:
                self.A_mat = self.mat_gauss(M, N_c, sfR)
        elif mat_type == 'dct':
            if self.algo_ori or self.algo_pa:
                self.A_ori = self.mat_dct_ori(M_all, N_all)
            if self.algo_sc:
                self.A_mat = self.mat_dct(M, N_c, sfR)
        else:
            raise ValueError("Type of sensing matrix does not exist!")

        self.noise_all_fc = noise_inst.noise_all_fc

        if self.algo_ori:
            self.x_all_ori = signal_inst.x_all_ori
            self.z_ori = self.A_ori @ self.x_all_ori
            self.y_ori = self.noise_all_fc(self.z_ori)

        if self.algo_pa:
            self.x_all_pa = signal_inst.x_all_pa
            self.Pl = signal_inst.Pl
            self.z_pa = self.A_ori @ self.x_all_pa
            self.y_pa = self.noise_all_fc(self.z_pa)

        if self.algo_sc:
            self.x_all_sc = signal_inst.x_all_sc
            self.x_vec, self.x_ra_vec = signal_inst.x_vec, signal_inst.x_ra_vec
            self.z_all = np.concatenate([self.A_mat[sfr] @ self.x_ra_vec[sfr] for sfr in range(sfR)], axis=0)
            # check1 = np.linalg.norm(self.z_all) ** 2 / M_all
            # self.y_all = self.z_all + self.n_all
            self.y_all = self.noise_all_fc(self.z_all)
            self.y_vec = self.y_vec_generate(self.y_all, M, sfR)

    def mat_gauss(self, M, N_c, sfR):
        A_mat = [np.zeros((M, N_c[sfr])) for sfr in range(sfR)]
        for sfr in range(sfR):
            A_mat[sfr] = np.random.normal(0, np.sqrt(1/N_c[sfr]), (M, N_c[sfr]))
        return A_mat

    def mat_gauss_ori(self, M_all, N_all):
        A_ori = np.random.normal(0, np.sqrt(1/N_all), (M_all, N_all))
        return A_ori

    def mat_dct(self, M, N_c, sfR):
        A_mat = [np.zeros((M, N_c[sfr])) for sfr in range(sfR)]
        for sfr in range(sfR):
            index_choice = np.random.choice(N_c[sfr], size=(M, ), replace=False)
            phase = np.random.choice([-1, 1], size=(N_c[sfr], ))
            A_dct = dct(np.eye(N_c[sfr]), norm='ortho')
            A_dct = phase * A_dct
            A_dct = idct(A_dct, norm='ortho')
            A_mat[sfr] = A_dct[index_choice, :].reshape(M, N_c[sfr])
        return A_mat

    def mat_dct_ori(self, M_all, N_all):
        index_choice = np.random.choice(N_all, size=(M_all, ), replace=False)
        phase = np.random.choice([-1, 1], size=(N_all, ))
        A_dct = dct(np.eye(N_all), norm='ortho')
        A_dct = phase * A_dct
        A_dct = idct(A_dct, norm='ortho')
        A_ori = A_dct[index_choice, :].reshape(M_all, N_all)
        return A_ori

    def y_vec_generate(self, y_all, M, sfR):
        y_vec = [y_all[M*sfr:M*(sfr+1)] for sfr in range(sfR)]
        return y_vec


class ParaComputation:
    def __init__(self, x_prior_type:str='bgauss', pa_type:str=None, mat_type:str="gauss", noise_type:str="awgnc",
                 M:int=1, N:int=1, B_src:int=1, Gamma:int=1, W:int=1, rho:float=1, delta:float=None, T:int=100,
                 snr:float=None, snrdb:float=None, sigma2:float=None, epsilon:float=None, zeta:float=1, N_se:int=1,
                 eps=1e-10, thre_vx_post:float=1e-20, rate:float=1, output_path:str=None, **kwargs) -> None:

        self.x_prior_type, self.pa_type, self.mat_type, self.noise_type = x_prior_type, pa_type, mat_type, noise_type
        self.N_se, self.delta, self.M, self.N, self.Gamma, self.W, self.T, self.rate = N_se, delta, M, N, Gamma, W, T, rate
        self.rho, self.B_src, self.zeta, self.snr, self.snrdb, self.sigma2, self.epsilon = rho, B_src, zeta, snr, snrdb, sigma2, epsilon
        self.eps, self.thre_vx_post, self.output_path = eps, thre_vx_post, output_path or "./output/"

        # noise settings
        if self.noise_type == 'awgnc':
            logging.info(f"Noise type: awgnc. Now setting noise power...")
            if self.snr is not None:
                self.sigma2 = 1 / self.snr
                self.snrdb = 10 * np.log10(self.snr)
                logging.info(f"Noise power: Prefer 1st to use snr.")
            elif self.snrdb is not None:
                self.sigma2 = np.power(10, - self.snrdb / 10)
                self.snr = np.power(10, self.snrdb / 10)
                logging.info(f"Noise power: Prefer 2nd to use snrdb.")
            else:
                self.snrdb = 10 * np.log10(1 / self.sigma2)
                self.snr = 1 / self.sigma2
                logging.info(f"Noise power: Prefer 3rd to use sigma2.")
        elif self.noise_type == 'bec':
            logging.info(f"Noise type: bec. Now setting noise power...")
            assert self.epsilon is not None, "The epsilon must be not None!"
            logging.info(f"Noise power: Prefer 1st to use epsilon.")
        else:
            raise ValueError("Type of noise does not exist!")

        # shape of sensing matrix
        self.sfC = self.Gamma
        self.sfR = self.Gamma + self.W - 1
        self.N_all = self.sfC * self.N
        if self.rate is not None:
            assert self.x_prior_type == 'src', "The prior of x must be src when using rate."
            # M_all_tmp = math.floor(self.N_all * np.log2(self.B_src) / (self.rate * self.B_src))
            M_all_tmp = (self.N_all * np.log2(self.B_src)) / (self.rate * self.B_src)
            self.M = math.floor(M_all_tmp / self.sfR)
            self.delta = self.M / self.N
            self.M_all = self.sfR * self.M
            logging.info(f"Matrix shape: Prefer 1st to use rate.")
        elif self.delta is not None:
            self.M = math.ceil(self.delta * self.N)
            self.M_all = self.sfR * self.M
            logging.info(f"Matrix shape: Prefer 2nd to use delta.")
        else:
            self.delta = self.M / self.N
            self.M_all = self.sfR * self.M
            logging.info(f"Matrix shape: Prefer 3rd to use M.")

        if self.x_prior_type == 'src':
            assert self.N % self.B_src == 0, "N must be a multiple of B_src!"
        assert self.M_all <= self.N_all, "M_all cannot be greater than N_all!"
        assert self.W <= self.Gamma, "W cannot be greater than Gamma!"

        # spatial coupling settings
        self.W_cal = [list(range(0, sfr+1)) for sfr in range(0, self.W-1)]
        self.W_cal.extend([list(range(sfr-self.W+1, sfr+1)) for sfr in range(self.W-1, self.Gamma)])
        self.W_cal.extend([list(range(sfr-self.W+1, self.Gamma)) for sfr in range(self.Gamma, self.Gamma+self.W-1)])

        self.N_c = [len(self.W_cal[sfr]) * self.N for sfr in range(self.sfR)]

        self.alpha = self.alpha_generate(self.M, self.N, self.sfR, self.sfC)
        self.alpha_c = [self.N_c[sfr] / self.M for sfr in range(self.sfR)]

    def alpha_generate(self, M, N, sfR, sfC):
        alpha = np.zeros((sfR, sfC))
        for sfr in range(sfR):
            for sfc in range(sfC):
                alpha[sfr, sfc] =  N / M
        return alpha


class OAMPConfiguration(ParaComputation):
    def __init__(self, parent_inst, signal_inst:SignalSettings, **kwargs) -> None:
        self.__dict__.update(parent_inst.__dict__)
        if 'mat_type' in kwargs:
            self.mat_type = kwargs['mat_type']

        self.algo_ori, self.algo_pa, self.algo_sc = signal_inst.algo_ori, signal_inst.algo_pa, signal_inst.algo_sc

        # channel settings
        noise_inst = NoiseSettings(noise_type=self.noise_type, M_all=self.M_all, sigma2=self.sigma2)
        coding_inst = CodingSettings(signal_inst=signal_inst, noise_inst=noise_inst, mat_type=self.mat_type,
                                      M_all=self.M_all, N_all=self.N_all, M=self.M, N_c=self.N_c, sfR=self.sfR)

        if self.algo_ori:
            self.lmmse_esti_ori = LMMSEEstiFunc(lmmse_type='vector')
            self.denoiser_bayes_ori = DenoiserBayesFunc(x_prior_type=self.x_prior_type, pa_type=None)
            self.x_prior_generate_ori = signal_inst.x_prior_generate_ori
            self.A_ori = coding_inst.A_ori
            self.x_all_ori, self.z_ori, self.y_ori = coding_inst.x_all_ori, coding_inst.z_ori, coding_inst.y_ori

        if self.algo_pa:
            self.lmmse_esti_pa = LMMSEEstiFunc(lmmse_type='vector')
            self.denoiser_bayes_pa = DenoiserBayesFunc(x_prior_type=self.x_prior_type, pa_type=self.pa_type)
            self.A_pa = coding_inst.A_ori
            self.Pl, self.x_all_pa, self.z_pa, self.y_pa = coding_inst.Pl, coding_inst.x_all_pa, coding_inst.z_pa, coding_inst.y_pa

        if self.algo_sc:
            self.lmmse_esti_sc = LMMSEEstiFunc(lmmse_type='vector')
            self.denoiser_bayes_sc = DenoiserBayesFunc(x_prior_type=self.x_prior_type, pa_type=None)
            self.x_prior_generate_sc = signal_inst.x_prior_generate_sc
            self.A_mat = coding_inst.A_mat
            self.x_all_sc, self.z_all, self.y_all = coding_inst.x_all_sc, coding_inst.z_all, coding_inst.y_all
            self.x_vec, self.y_vec, self.x_ra_vec = coding_inst.x_vec, coding_inst.y_vec, coding_inst.x_ra_vec

    def config_print(self):
        logging.info(f"Configuration Parameters:")
        logging.info(f"mat_type: {self.mat_type} | x_prior_type: {self.x_prior_type} | noise_type: {self.noise_type} | "
                     f"pa_type: {'None' if self.pa_type == None else self.pa_type} | "
                     f"T: {self.T} | N_se: {self.N_se} | M: {self.M} | N: {self.N} | delta: {self.delta:.4f} | "
                     f"Gamma: {self.Gamma} | W: {self.W} | zeta: {self.zeta:.3f} | "
                     f"B_src: {'None' if self.x_prior_type != 'src' else self.B_src} | "
                     f"rho: {'None' if self.x_prior_type != 'bgauss' else f'{self.rho:.4f}'} | "
                     f"sigma2: {'None' if self.noise_type != 'awgnc' else f'{self.sigma2:.4f}'} | "
                     f"snrdb: {'None' if self.noise_type != 'awgnc' else f'{self.snrdb:.4f}'} | "
                     f"snr: {'None' if self.noise_type != 'awgnc' else f'{self.snr:.4f}'} | "
                     f"epsilon: {'None' if self.noise_type not in ['bec', 'bsc'] else f'{self.epsilon:.4f}'} | "
                     f"M_all: {self.M_all} | N_all: {self.N_all} | rate: {self.rate:.4f} | "
                     f"Overall compression rate: {self.M_all/self.N_all:.3f}")
        # logging.info(f"Successful to generate configuration!")
