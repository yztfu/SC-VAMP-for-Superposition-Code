import os
import logging
import datetime
import argparse
import numpy as np
from oamp.oamp_ori import iteration_oamp_ori
from oamp.oamp_pa import iteration_oamp_pa
from oamp.oamp_sc import iteration_oamp_sc
from oamp.utils import SignalSettings, OAMPConfiguration, ParaComputation


def parse_args():
    parser = argparse.ArgumentParser(description='OAMP for spatial coupling.')

    parser.add_argument('--algo_ori', action='store_true', help='whether use oamp_ori.')
    parser.add_argument('--algo_pa', action='store_true', help='whether use oamp_pa.')
    parser.add_argument('--algo_sc', action='store_true', help='whether use oamp_sc.')

    parser.add_argument('--x_prior_type', type=str, default='src', help='Type of prior of signal.')
    parser.add_argument('--pa_type', type=str, default=None, help='Type of power allocation.')

    parser.add_argument('--mat_type', type=str, default='dct', help='Type of sensing matrix.')
    parser.add_argument('--noise_type', type=str, default='awgnc', help='Type of adding noise.')

    parser.add_argument('--N_se', type=int, default=10000, help='value of N_se.')
    parser.add_argument('--M', type=int, default=256, help='value of M.')
    parser.add_argument('--N', type=int, default=512, help='value of N.')
    parser.add_argument('--B_src', type=int, default=4, help='value of B_src.')
    parser.add_argument('--Gamma', type=int, default=16, help='value of Gamma.')
    parser.add_argument('--W', type=int, default=1, help='value of W.')
    parser.add_argument('--zeta', type=float, default=0.7, help='value of damping factor.')
    parser.add_argument('--snrdb', type=float, default=10, help='value of signal to noise rate (dB).')
    parser.add_argument('--snr', type=float, default=10, help='value of signal to noise rate.')
    parser.add_argument('--sigma2', type=float, default=1e-3, help='value of signal power.')
    parser.add_argument('--rho', type=float, default=0.1, help='value of rho.')
    parser.add_argument('--T', type=int, default=200, help='Iteration times.')
    parser.add_argument('--delta', type=float, default=0.4, help='value of delta.')
    parser.add_argument('-R', '--rate', type=float, default=1.4, help='value of rate.')

    parser.add_argument('--num_trials', type=int, default=50, help='value of number of trails.')
    parser.add_argument('--save_interval', type=int, default=5, help='value of save interval.')
    parser.add_argument('--output_path', type=str, default='./output/', help='path of output data.')

    args = parser.parse_args()

    return args


def _print_args(title, args):
    logging.info(f"---------------- {title} ----------------")
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append(f"  {arg} {dots} {getattr(args, arg)}")
    for arg in sorted(str_list, key=lambda x: x.lower()):
        logging.info(arg)
    logging.info(f"---------------- end of {title} ----------------")
    logging.info(f"\n")


def signal_generate(args, para:ParaComputation):
    logging.info(f"==== Generating signal... ====")
    signal_inst = SignalSettings(algo_ori=args.algo_ori, algo_pa=args.algo_pa, algo_sc=args.algo_sc,
                                  x_prior_type=para.x_prior_type, pa_type=para.pa_type, N_all=para.N_all, sigma2=para.sigma2, rho=para.rho,
                                  B_src=para.B_src, N=para.N, N_c=para.N_c, W_cal=para.W_cal, sfR=para.sfR, sfC=para.sfC)
    logging.info(f"==== Generate signal successfully! ====")
    return signal_inst


def para_computation_generate(args):
    logging.info(f"==== Generating parameters computation... ====")
    para = ParaComputation(x_prior_type=args.x_prior_type, pa_type=args.pa_type, mat_type=args.mat_type, noise_type=args.noise_type,
                            M=args.M, N=args.N, B_src=args.B_src, Gamma=args.Gamma, W=args.W, T=args.T, delta=args.delta,
                            snrdb=args.snrdb, sigma2=args.sigma2, snr=args.snr, rho=args.rho, zeta=args.zeta,
                            N_se=args.N_se, rate=args.rate, output_path=args.output_path)
    logging.info(f"==== Generate parameters computation successfully! ====")
    return para


def configuration_generate(para:ParaComputation, signal_inst:SignalSettings, mat_type:str='gauss'):
    logging.info(f"==== Generating oamp configuration... ====")
    config = OAMPConfiguration(para, signal_inst, mat_type=mat_type)
    logging.info(f"==== Generate oamp configuration successfully! ====")
    return config


if __name__ == "__main__":
    # arguments parse
    args = parse_args()

    # logging settings
    now = datetime.datetime.now()
    timestamp = now.strftime("%y-%m-%d_%H:%M:%S")
    log_path = args.output_path + f"log/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + f"oamp_all_B{args.B_src}_R{args.rate:.2f}_W{args.W}_{timestamp}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_name),
            logging.StreamHandler()
        ]
    )

    # print arguments
    _print_args("arguments", args)
    if args.mat_type == 'all':
        mat_type_all = ['dct', 'gauss']
    else:
        mat_type_all = [args.mat_type, ]
    nums_mat_type = len(mat_type_all)

    ser_sc_gt = np.zeros((args.num_trials, nums_mat_type, args.Gamma, args.T))
    mse_sc_gt = np.zeros((args.num_trials, nums_mat_type, args.Gamma, args.T))
    mse_sc_esti = np.zeros((args.num_trials, nums_mat_type, args.Gamma, args.T))
    ser_sc_se = np.zeros((args.num_trials, nums_mat_type, args.Gamma, args.T))
    mse_sc_se = np.zeros((args.num_trials, nums_mat_type, args.Gamma, args.T))

    ser_ori_gt = np.zeros((args.num_trials, nums_mat_type, 1, args.T))
    mse_ori_gt = np.zeros((args.num_trials, nums_mat_type, 1, args.T))
    mse_ori_esti = np.zeros((args.num_trials, nums_mat_type, 1, args.T))
    ser_ori_se = np.zeros((args.num_trials, nums_mat_type, 1, args.T))
    mse_ori_se = np.zeros((args.num_trials, nums_mat_type, 1, args.T))

    ser_pa_gt = np.zeros((args.num_trials, nums_mat_type, 1, args.T))
    mse_pa_gt = np.zeros((args.num_trials, nums_mat_type, 1, args.T))
    mse_pa_esti = np.zeros((args.num_trials, nums_mat_type, 1, args.T))
    ser_pa_se = np.zeros((args.num_trials, nums_mat_type, 1, args.T))
    mse_pa_se = np.zeros((args.num_trials, nums_mat_type, 1, args.T))

    for i in range(args.num_trials):
        logging.info(f"------------------------ trials {i+1:03d} ------------------------\n")
        para = para_computation_generate(args=args)
        signal_inst = signal_generate(args=args, para=para)

        for j in range(len(mat_type_all)):
            logging.info(f"\n")
            config = configuration_generate(para=para, signal_inst=signal_inst, mat_type=mat_type_all[j])
            config.config_print()
            # ser_ori_gt[i, j], mse_ori_gt[i, j], mse_ori_esti[i, j], ser_ori_se[i, j], mse_ori_se[i, j] = iteration_oamp_ori(config=config)
            # ser_pa_gt[i, j], mse_pa_gt[i, j], mse_pa_esti[i, j], ser_pa_se[i, j], mse_pa_se[i, j] = iteration_oamp_pa(config=config)
            ser_sc_gt[i, j], mse_sc_gt[i, j], mse_sc_esti[i, j], ser_sc_se[i, j], mse_sc_se[i, j] = iteration_oamp_sc(config=config)

        if i == args.num_trials - 1 or (i+1) % args.save_interval == 0:
            logging.info(f"==== saving data... ====")
            file_path = args.output_path + f"data/"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file_name = file_path + f"oamp_all_B{config.B_src}_R{config.rate:.2f}_W{config.W}.npz"
            np.savez(file_name, ser_sc_gt=ser_sc_gt, mse_sc_gt=mse_sc_gt, mse_sc_esti=mse_sc_esti, ser_sc_se=ser_sc_se, mse_sc_se=mse_sc_se,
                    ser_ori_gt=ser_ori_gt, mse_ori_gt=mse_ori_gt, mse_ori_esti=mse_ori_esti, ser_ori_se=ser_ori_se, mse_ori_se=mse_ori_se,
                    ser_pa_gt=ser_pa_gt, mse_pa_gt=mse_pa_gt, mse_pa_esti=mse_pa_esti, ser_pa_se=ser_pa_se, mse_pa_se=mse_pa_se)
            logging.info(f"==== save data successfully! ====")
        logging.info(f"\n\n")
