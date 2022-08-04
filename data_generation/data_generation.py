import pandas as pd
import numpy as np

from time import perf_counter
from egret.parsers.matpower_parser import create_ModelData
from egret.models.acopf import solve_acopf, create_psv_acopf_model


def mvnrnd_trn_gibbs(lb, ub, mu, sigMat, NUM_smp):
    """Sample truncated multivariate normal distribution using Gibbs Sampler

    Parameters
    ----------
    lb
        lower bound vector [1, d]
    ub
        upper bound vector [1, d]
    mu
        means vector [1, d]
    sigMat
        covariance matrix [d, d]
    NUM_smp
        number of samples

    Returns
    -------
    smpOut
        matrix of all samples [NUM_smp, d]
    """

    k = np.size(lb)

    # The linear constraint (B*x <= b)
    B = np.r_[np.eye(k), -np.eye(k)]
    b = np.r_[ub, -lb]
    b = b.reshape(b.shape[0], 1)

    S, V = np.linalg.eig(sigMat)
    A = np.lib.scimath.sqrt(np.linalg.inv(V)) * S.conj().T
    A = np.array(A)

    # Define new variable z = A*x, z~Nt(A*mu, I), T={z: D*z<=b}
    mu_ = A@mu
    D = np.linalg.lstsq(A.conj().T, np.array(B).conj().T, rcond=None)[0].conj().T

    rng = np.random.default_rng()
    x = np.empty((k, NUM_smp), dtype="complex_")
    x.fill(np.nan)
    for ii in range(k):
        if not ub[ii] - lb[ii] < 0:
            x[ii, 0] = rng.uniform(lb[ii], ub[ii])
        else:
            x[ii, 0] = rng.uniform(ub[ii], lb[ii])

    Z = np.empty((k, NUM_smp), dtype="complex_")
    Z.fill(np.nan)

    Z[:, 0] = A@x[:, 0]

    dims = np.arange(k)  # INDEX INTO EACH DIMENSION
    # RUN GIBBS SAMPLER
    t = 0
    while t < NUM_smp:
        for iD in dims:  # LOOP OVER DIMENSIONS
            count = 1
            # UPDATE SAMPLES
            nIx = dims != iD  # *NOT* THE CURRENT DIMENSION
            # CONDITIONAL MEAN
            muCond = mu_[iD]
            # GENERATE SAMPLE
            Z[iD, t] = muCond + rng.normal()
            # CHECK CONSTRAINT
            di = D[:, iD]
            z_i = np.r_[Z[:iD, t], Z[iD+1:, t-1]]
            D_i = D[:, nIx]

            left_part = np.array(di*Z[iD, t])
            left_part = left_part.reshape(left_part.shape[0], 1)
            while np.all(np.sum(np.greater(left_part, b - D_i*z_i), axis=1)):
                if count == 11:
                    lb_idx = di < 0
                    lb_i = np.min((b[lb_idx] - D_i[lb_idx, :] @ z_i) / di[lb_idx])
                    ub_idx = di > 0
                    ub_i = np.max((b[ub_idx] - D_i[ub_idx, :] @ z_i) / di[ub_idx])
                    real_rand = rng.uniform(np.real(lb_i), np.real(ub_i))
                    imag_rand = 1j*rng.uniform(np.imag(lb_i), np.imag(ub_i))
                    complex_rand = real_rand + imag_rand
                    Z[iD, t] = complex_rand
                else:
                    Z[iD, t] = muCond + rng.normal()
                    count += 1

                left_part = np.array(di*Z[iD, t])
                left_part = left_part.reshape(left_part.shape[0], 1)

        x[:, t] = np.linalg.lstsq(A, Z[:, t], rcond=None)[0]
        t += 1

    return x


def generate_samples(md, NSamples: int, NL, CorrCoeff=0.4, MAX_PF=0.8, MIN_PF=0.1,
                     MaxChangeLoad=0.7):
    """Generates load demand samples utilizng the gibbs sampler algorithm and returns them

    Parameters
    ----------
    md : Any
        ModelData object of the network with its base loading profile
    NSamples : int
        The numeber of samples that are to be generated
    NL : int
        The number of load busses in the network
    CorrCoeff : float, default = 0.4
        Used to calculate the Covariance Matrix
    MAX_PF : float, default = 0.8

    MIN_PF : float, default = 0.1

    """
    loads = dict(md.elements(element_type="load"))

    p_loads = [np.fix(loads[key]["p_load"]) for key in loads.keys()]
    q_loads = [np.fix(loads[key]["q_load"]) for key in loads.keys()]

    pq_loads = np.stack((p_loads, q_loads), axis=1)

    # Generate load samples
    ScaleSigma = MaxChangeLoad/1.645

    mu = pq_loads[:, 0]
    sigma = ScaleSigma**2 * (CorrCoeff * pq_loads[:, 0]) + ((1-CorrCoeff) * np.eye(NL))

    LoadFactor = mvnrnd_trn_gibbs((1-MaxChangeLoad)*pq_loads[:, 0],
                                  (1+MaxChangeLoad)*pq_loads[:, 0],
                                  mu,
                                  sigma,
                                  NSamples)

    rng = np.random.default_rng()
    PFFactor = (rng.uniform(size=(NL, NSamples)) * (MIN_PF - MAX_PF)) + MAX_PF

    # Use the real part of the complex number number for the active power load
    gen_p_loads = [list(np.real(load)) for load in LoadFactor]
    gen_q_loads = [list(load * factor) for load, factor in zip(gen_p_loads, PFFactor)]

    return gen_p_loads, gen_q_loads


def solve_acopf_return_data(md, p_load, q_load):
    kwargs = {'include_feasibility_slack': False}

    try:
        md, results = solve_acopf(model_data, "ipopt", solver_tee=False,
                                  acopf_model_generator=create_psv_acopf_model,
                                  return_results=True, **kwargs)
    except:
        return None

    p_generated = [
        md.data['elements']['generator'][key]['pg']
        for key in md.data['elements']['generator'].keys()]
    q_generated = [
        md.data['elements']['generator'][key]['qg']
        for key in md.data['elements']['generator'].keys()]
    v_generated = [
        md.data['elements']['generator'][key]['vg']
        for key in md.data['elements']['generator'].keys()]

    df_cols = {}
    df_cols['p_loads'] = {
        'p_load{}'.format(i + 1): p_load for p_load,
        i in zip(p_load, range(len(q_load)))}
    df_cols['q_loads'] = {
        'q_load{}'.format(i + 1): q_load for q_load,
        i in zip(p_load, range(len(q_load)))}
    df_cols['p_gens'] = {'p_gen{}'.format(i + 1): p_gen for p_gen,
                         i in zip(p_generated, range(len(p_generated)))}
    df_cols['q_gens'] = {'q_gen{}'.format(i + 1): q_gen for q_gen,
                         i in zip(q_generated, range(len(q_generated)))}
    df_cols['v_gens'] = {'v_gen{}'.format(i + 1): v_gen for v_gen,
                         i in zip(v_generated, range(len(v_generated)))}
    df_cols['total_cost'] = md.data['system']['total_cost']
    df_cols['time_to_solve'] = results['Solver'][0]['Time']

    cols = {inner_key: [] for key in df_cols.keys() if key != 'total_cost' and key !=
            'time_to_solve' for inner_key in df_cols[key].keys()}
    cols['total_cost'] = 0
    cols['time_to_solve'] = 0

    df = pd.DataFrame(cols)

    for key in df_cols.keys():
        if key != 'total_cost' and key != 'time_to_solve':
            for inner_key in df_cols[key].keys():
                df[inner_key] = [df_cols[key][inner_key]]

    df['total_cost'] = df_cols['total_cost']
    df['time_to_solve'] = df_cols['time_to_solve']

    return df


if __name__ == "__main__":
    original_case118 = "/Users/nogueras1/Documents/ACOPF_Workspace/data_generation/case118_data/case118.txt"

    tic = perf_counter()

    NSamples = 100000
    successful_samples = 0
    df = pd.DataFrame()
    while successful_samples < NSamples:
        model_data = create_ModelData(original_case118)
        num_of_loads = len(model_data.data['elements']['load'].keys())

        gen_tic = perf_counter()
        gen_p_loads, gen_q_loads = generate_samples(model_data, NSamples, num_of_loads)
        gen_toc = perf_counter()

        print(
            "It took {} seconds to generate {} samples, now we solve them".format(
                gen_toc - gen_tic, NSamples))

        solve_500_tic = perf_counter()
        for idx in range(len(gen_p_loads[0])):
            for p_load, q_load, key in zip(
                    gen_p_loads, gen_q_loads, model_data.data['elements']['load'].keys()):
                model_data.data['elements']['load'][key]['p_load'] = p_load[idx]
                model_data.data['elements']['load'][key]['q_load'] = q_load[idx]

            data = solve_acopf_return_data(
                model_data, np.asarray(gen_p_loads)[:, idx],
                np.asarray(gen_q_loads)[:, idx])

            if data is not None:
                df = pd.concat([df, data])
                successful_samples += 1

            if successful_samples % 100 == 0:
                df.to_csv("case118_data4.csv")

            if successful_samples % 500 == 0:
                solve_500_toc = perf_counter()
                print("It took {} seconds to generate 500 samples; {} samples in total".format(
                    solve_500_toc - solve_500_tic, successful_samples))

        if successful_samples < NSamples:
            NSamples = NSamples - successful_samples
            print("We still need to generate {} successful samples".format(NSamples))
            if NSamples < 100:
                NSamples = 100

    toc = perf_counter()

    print("Time to generate {} samples : {}".format(successful_samples, toc - tic))
    df.to_csv("case118_data4.csv")
