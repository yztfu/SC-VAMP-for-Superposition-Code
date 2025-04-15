import numpy as np
import copy
import logging
from .utils import OAMPConfiguration


def oamp_sc_linear(config:OAMPConfiguration, r1_ra:list[np.ndarray], gamma1:list[float]):
    r1_ra_pre = copy.deepcopy(r1_ra)
    gamma1_pre = copy.deepcopy(gamma1)

    x1_ra = [np.zeros((config.N_c[sfr], 1)) for sfr in range(config.sfR)]
    alpha1 = [0, ] * config.sfR
    eta1 = [0, ] * config.sfR
    r2_ra = [np.zeros((config.N_c[sfr], 1)) for sfr in range(config.sfR)]
    gamma2 = [0, ] * config.sfR
    for sfr in range(config.sfR):
        x1_ra[sfr], alpha1[sfr] = config.lmmse_esti_sc(r1=r1_ra[sfr], y=config.y_vec[sfr], gamma1=gamma1[sfr], snr=config.snr, A_mat=config.A_mat[sfr])
        eta1[sfr] = gamma1[sfr] / alpha1[sfr]
        gamma2[sfr] = eta1[sfr] - gamma1[sfr]
        r2_ra[sfr] = (eta1[sfr] * x1_ra[sfr] - gamma1[sfr] * r1_ra[sfr]) / gamma2[sfr]

    r2_hat = [np.zeros((config.N, 1)) for _ in range(config.sfC)]
    gamma2_hat = [0, ] * config.sfC
    x_post = [np.zeros((config.N, 1)) for _ in range(config.sfC)]
    v_post = [0, ] * config.sfC
    x2_hat = [np.zeros((config.N, 1)) for _ in range(config.sfC)]
    alpha2_hat = [0, ] * config.sfC
    eta2_hat = [0, ] * config.sfC
    for sfc in range(config.sfC):
        gamma2_hat[sfc] = np.sum(np.array(gamma2[sfc:sfc+config.W]))
        r2_hat_w = np.zeros((config.N, config.W))
        for w in range(config.W):
            if sfc + w <= config.W - 1:
                w_cal = len(config.W_cal[sfc+w]) - 1 - w
            else:
                w_cal = config.W - 1 - w
            r2_hat_w[:, w:w+1] = r2_ra[sfc+w][w_cal*config.N:(w_cal+1)*config.N]  # shape: (N, W)
        r2_hat[sfc] = (np.sum(np.array(gamma2[sfc:sfc+config.W]) * r2_hat_w, axis=1) / gamma2_hat[sfc]).reshape(-1, 1)  # sum((N, W), axis=1)

        x_post[sfc], v_post[sfc] = config.denoiser_bayes_sc(beta_pri=r2_hat[sfc], var_pri=(1/gamma2_hat[sfc]), rho=config.rho, N_all=config.N, B_src=config.B_src, eps=config.eps)
        x2_hat[sfc], alpha2_hat[sfc], eta2_hat[sfc] = x_post[sfc], (gamma2_hat[sfc] * v_post[sfc]), (1 / v_post[sfc])

    x2_ra = [np.zeros((config.N_c[sfr], 1)) for sfr in range(config.sfR)]
    eta2 = [0, ] * config.sfR
    r1_ra = [np.zeros((config.N_c[sfr], 1)) for sfr in range(config.sfR)]
    gamma1 = [0, ] * config.sfR
    for sfr in range(config.sfR):
        x2_ra[sfr] = np.concatenate([x2_hat[w] for w in config.W_cal[sfr]], axis=0)
        eta2[sfr] = len(config.W_cal[sfr]) / np.sum([1 / eta2_hat[w] for w in config.W_cal[sfr]])
        gamma1[sfr] = eta2[sfr] - gamma2[sfr]
        r1_ra[sfr] = (eta2[sfr] * x2_ra[sfr] -  gamma2[sfr] * r2_ra[sfr]) / gamma1[sfr]

        # damping factor
        gamma1[sfr] = 1 / (config.zeta / gamma1[sfr] + (1 - config.zeta) / gamma1_pre[sfr])
        r1_ra[sfr] = config.zeta * r1_ra[sfr] + (1 - config.zeta) * r1_ra_pre[sfr]

    return r1_ra, gamma1, x_post, v_post


def se_oamp_sc_linear(config:OAMPConfiguration, gamma1:list[float]):
    alpha1 = [0, ] * config.sfR
    eta1 = [0, ] * config.sfR
    gamma2 = [0, ] * config.sfR
    for sfr in range(config.sfR):
        alpha1[sfr] = config.lmmse_esti_sc(gamma1=gamma1[sfr], snr=config.snr, A_mat=config.A_mat[sfr], se=True)
        eta1[sfr] = gamma1[sfr] / alpha1[sfr]
        gamma2[sfr] = eta1[sfr] - gamma1[sfr]

    r2x_hat = [np.zeros((config.N_se, 1)) for _ in range(config.sfC)]
    r2z_hat = [np.zeros((config.N_se, 1)) for _ in range(config.sfC)]
    r2_hat = [np.zeros((config.N_se, 1)) for _ in range(config.sfC)]
    gamma2_hat = [0, ] * config.sfC
    x_post = [np.zeros((config.N_se, 1)) for _ in range(config.sfC)]
    v_post = [0, ] * config.sfC
    x2_hat = [np.zeros((config.N_se, 1)) for _ in range(config.sfC)]
    alpha2_hat = [0, ] * config.sfC
    eta2_hat = [0, ] * config.sfC
    for sfc in range(config.sfC):
        gamma2_hat[sfc] = np.sum(np.array(gamma2[sfc:sfc+config.W]))
        # online average of x
        r2x_hat[sfc] = config.x_prior_generate_sc(N_all=config.N_se, rho=config.rho, B_src=config.B_src)
        r2z_hat[sfc] = np.random.normal(0, np.sqrt(1/gamma2_hat[sfc]), (config.N_se, 1))
        r2_hat[sfc] = r2x_hat[sfc] + r2z_hat[sfc]

        x_post[sfc], v_post[sfc] = config.denoiser_bayes_sc(beta_pri=r2_hat[sfc], var_pri=(1/gamma2_hat[sfc]), rho=config.rho, N_all=config.N_se, B_src=config.B_src, eps=config.eps)
        x2_hat[sfc], alpha2_hat[sfc], eta2_hat[sfc] = x_post[sfc], (gamma2_hat[sfc] * v_post[sfc]), 1 / v_post[sfc]

    eta2 = [0, ] * config.sfR
    gamma1 = [0, ] * config.sfR
    for sfr in range(config.sfR):
        eta2[sfr] = len(config.W_cal[sfr]) / np.sum([1 / eta2_hat[w] for w in config.W_cal[sfr]])
        gamma1[sfr] = eta2[sfr] - gamma2[sfr]

    return gamma1, v_post, r2x_hat, x_post


def oamp_sc_initialization(config:OAMPConfiguration):
    r1_ra = [np.zeros((config.N_c[sfr], 1)) for sfr in range(config.sfR)]
    gamma1 = [1, ] * config.sfR
    return r1_ra, gamma1


def se_oamp_sc_initialization(config:OAMPConfiguration):
    gamma1 = [1, ] * config.sfR
    return gamma1


def ga_calculation_list(x_post, x_gt, B_src=1):
    L = len(x_gt)
    ser_gt = np.zeros(L)
    mse_gt = np.zeros(L)
    for l in range(L):
        N = x_gt[l].shape[0]
        L_src = N // B_src
        errors = 0
        for i in range(L_src):
            idx_post = np.argmax(x_post[l][i*B_src:(i+1)*B_src])
            idx_gt = np.argmax(x_gt[l][i*B_src:(i+1)*B_src])
            errors += int(idx_post != idx_gt)
        ser_gt[l] = errors / L_src
        mse_gt[l] = np.sum((x_post[l] - x_gt[l]) ** 2) / N
    return ser_gt, mse_gt


def output_from_error_elem(error, name):
    error_max, error_max_index = np.max(error), np.argmax(error)
    error_min, error_min_index = np.min(error), np.argmin(error)
    error_mean = np.mean(error)
    str_head = " "*(10 - len(name)) + f"{name}"
    str_all = str_head + f"  *ALL*"
    len_n = 8
    for i in range(len(error)):
        str_all += f" | {i:02d}: {error[i]:.2e}"
        if (i+1) % len_n == 0 and i != len(error) - 1:
            str_all += f"\n" + " " * (len(str_head) + 7)
    logging.info(str_all)
    logging.info(" " * len(str_head) + f"  *MAX* | value: {error_max:.4e}, index: {error_max_index:02d}")
    logging.info(" " * len(str_head) + f"  *MIN* | value: {error_min:.4e}, index: {error_min_index:02d}")
    logging.info(" " * len(str_head) + f" *MEAN* | value: {error_mean:.4e}")


def output_from_error_all(dt, ser_gt, mse_gt, mse_esti, ser_se, mse_se):
    logging.info(f"*** Iteration: {dt:03d} ***")
    output_from_error_elem(ser_gt, 'ser_gt')
    output_from_error_elem(mse_gt, 'mse_gt')
    output_from_error_elem(mse_esti, 'mse_esti')
    output_from_error_elem(ser_se, 'ser_se')
    output_from_error_elem(mse_se, 'mse_se')


def output_from_error_algo(dt, ser_gt, mse_gt, mse_esti):
    logging.info(f"*** Iteration: {dt:03d} ***")
    output_from_error_elem(ser_gt, 'ser_gt')
    output_from_error_elem(mse_gt, 'mse_gt')
    output_from_error_elem(mse_esti, 'mse_esti')


def output_from_error_se(dt, ser_se, mse_se):
    logging.info(f"*** Iteration: {dt:03d} ***")
    output_from_error_elem(ser_se, 'ser_se')
    output_from_error_elem(mse_se, 'mse_se')


def iteration_oamp_sc(config:OAMPConfiguration):
    '''
    The main iteration of oamp_sc
    '''
    # Print configuration
    # config.config_print()
    print(f"oamp_sc iterations start...")
    # Initialization
    r1_ra, gamma1 = oamp_sc_initialization(config=config)
    gamma1_se = se_oamp_sc_initialization(config=config)

    ser_gt_ite = np.zeros((config.Gamma, config.T))
    mse_gt_ite = np.zeros((config.Gamma, config.T))
    mse_esti_ite = np.zeros((config.Gamma, config.T))
    ser_se_ite = np.zeros((config.Gamma, config.T))
    mse_se_ite = np.zeros((config.Gamma, config.T))
    # Iterations
    for dt in range(config.T):
        r1_ra, gamma1, x_post, v_post = oamp_sc_linear(config, r1_ra, gamma1)
        ser_gt, mse_gt = ga_calculation_list(x_post, config.x_vec, config.B_src)
        mse_esti = np.array(v_post)
        ser_gt_ite[:, dt] = ser_gt
        mse_gt_ite[:, dt] = mse_gt
        mse_esti_ite[:, dt] = mse_esti

        gamma1_se, v_post_se, x_vec_se, x_post_se = se_oamp_sc_linear(config, gamma1_se)
        ser_se, _ = ga_calculation_list(x_post_se, x_vec_se, config.B_src)
        mse_se = np.array(v_post_se)
        ser_se_ite[:, dt] = ser_se
        mse_se_ite[:, dt] = mse_se

        output_from_error_all(dt, ser_gt, mse_gt, mse_esti, ser_se, mse_se)
        # output_from_error_algo(dt, ser_gt, mse_gt, mse_esti)

    print(f"oamp_sc iterations end.\n")

    return ser_gt_ite, mse_gt_ite, mse_esti_ite, ser_se_ite, mse_se_ite
