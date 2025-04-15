import numpy as np
import logging
from .utils import OAMPConfiguration


def oamp_ori_linear(config:OAMPConfiguration, r1:np.ndarray, gamma1:float):
    x1, alpha1 = config.lmmse_esti_ori(r1=r1, y=config.y_ori, gamma1=gamma1, snr=config.snr, A_mat=config.A_ori)
    eta1 = gamma1 / alpha1
    gamma2 = eta1 - gamma1
    r2 = (eta1 * x1 - gamma1 * r1) / gamma2

    x_post, v_post = config.denoiser_bayes_ori(beta_pri=r2, var_pri=(1/gamma2), rho=config.rho, N_all=config.N_all, B_src=config.B_src, eps=config.eps)
    x2, alpha2 = x_post, (gamma2 * v_post)
    eta2 = gamma2 / alpha2
    gamma1 = eta2 - gamma2
    r1 = (eta2 * x2 - gamma2 * r2) / gamma1

    return r1, gamma1, x_post, v_post


def se_oamp_ori_linear(config:OAMPConfiguration, gamma1:float):
    alpha1 = config.lmmse_esti_ori(gamma1=gamma1, snr=config.snr, A_mat=config.A_ori, se=True)
    eta1 = gamma1 / alpha1
    gamma2 = eta1 - gamma1

    # online average of x
    r2x = config.x_prior_generate_ori(N_all=config.N_se, rho=config.rho, B_src=config.B_src, sigma2=config.sigma2)
    r2n = np.random.normal(0, np.sqrt(1/gamma2), (config.N_se, 1))
    r2 = r2x + r2n
    x_post, v_post = config.denoiser_bayes_ori(beta_pri=r2, var_pri=(1/gamma2), rho=config.rho, N_all=config.N_se, B_src=config.B_src, eps=config.eps)

    alpha2 = gamma2 * v_post
    eta2 = gamma2 / alpha2
    gamma1 = eta2 - gamma2

    return gamma1, v_post, r2x, x_post


def oamp_initialization(config:OAMPConfiguration):
    r1 = np.zeros((config.N_all, 1))
    gamma1 = 1
    return r1, gamma1


def se_oamp_initialization(config:OAMPConfiguration):
    gamma1 = 1
    return gamma1


def ga_calculation(x_post, x_gt, B_src=1):
    N = x_gt.shape[0]
    mse_gt = np.sum((x_post - x_gt) ** 2) / N

    L_src = N // B_src
    errors = 0
    for i in range(L_src):
        idx_post = np.argmax(x_post[i*B_src:(i+1)*B_src])
        idx_gt = np.argmax(x_gt[i*B_src:(i+1)*B_src])
        errors += int(idx_post != idx_gt)
    ser_gt = errors / L_src
    return ser_gt, mse_gt


def output_from_error_all(dt, ser_gt, mse_gt, mse_esti, ser_se, mse_se):
    logging.info(f"*** Iteration: {dt:03d} ***")
    logging.info(f"   *MEAN* | ser_gt: {ser_gt:.4e} | mse_gt: {mse_gt:.4e} | "
                 f"mse_esti: {mse_esti:.4e} | ser_se: {ser_se:.4e} | mse_se: {mse_se:.4e}")


def output_from_error_algo(dt, ser_gt, mse_gt, mse_esti):
    logging.info(f"*** Iteration: {dt:03d} ***")
    logging.info(f"   *MEAN* | ser_gt: {ser_gt:.4e} | mse_gt: {mse_gt:.4e} | mse_esti: {mse_esti:.4e}")


def output_from_error_se(dt, ser_se, mse_se):
    logging.info(f"*** Iteration: {dt:03d} ***")
    logging.info(f"   *MEAN* | ser_se: {ser_se:.4e} | mse_se: {mse_se:.4e}")


def iteration_oamp_ori(config:OAMPConfiguration):
    '''
    The main iteration of oamp_ori
    '''
    # Print configuration
    # config.config_print()
    logging.info(f"oamp_ori iterations start...")
    # Initialization
    r1, gamma1 = oamp_initialization(config=config)
    gamma1_se = se_oamp_initialization(config=config)

    ser_gt_ite = np.zeros((1, config.T))
    mse_gt_ite = np.zeros((1, config.T))
    mse_esti_ite = np.zeros((1, config.T))
    ser_se_ite = np.zeros((1, config.T))
    mse_se_ite = np.zeros((1, config.T))
    # Iterations
    for dt in range(config.T):
        r1, gamma1, x_post, v_post = oamp_ori_linear(config, r1, gamma1)
        ser_gt, mse_gt = ga_calculation(x_post, config.x_all_ori, config.B_src)
        mse_esti = v_post
        ser_gt_ite[:, dt] = ser_gt
        mse_gt_ite[:, dt] = mse_gt
        mse_esti_ite[:, dt] = mse_esti

        gamma1_se, v_post_se, x_all_se, x_post_se = se_oamp_ori_linear(config, gamma1_se)
        ser_se, _ = ga_calculation(x_post_se, x_all_se, config.B_src)
        mse_se = v_post_se
        ser_se_ite[:, dt] = ser_se
        mse_se_ite[:, dt] = mse_se

        output_from_error_all(dt, ser_gt, mse_gt, mse_esti, ser_se, mse_se)
        # output_from_error_algo(dt, ser_gt, mse_gt, mse_esti)

    logging.info(f"oamp_ori iterations end.\n")

    return ser_gt_ite, mse_gt_ite, mse_esti_ite, ser_se_ite, mse_se_ite
