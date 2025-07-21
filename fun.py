import numpy as np


def sim_LIF(options):

    # extract
    dt = options['dt']
    t_total = options['t_total']
    I_stim_on = options['I_stim_on']
    I_stim_off = options['I_stim_off']
    I_stim = options['I_stim']
    V_L = options['V_L']
    tau_V = options['tau_V']
    R = options['R']
    V_th = options['V_th']
    V_r = options['V_r']
    tau_ref = options['tau_ref']

    # initialize
    t_vec = np.array([i * dt for i in range(int(t_total / dt))])
    I_vec = np.zeros(len(t_vec))
    V_vec = np.zeros(len(t_vec))
    spktimes = []
    ref = -1
    V = V_L

    # loop over time
    for i, t in enumerate(t_vec):
        
        # update input
        if (t >= I_stim_on) and (t <= I_stim_off):
            I = I_stim
        else:
            I = 0
        I_vec[i] = I
        
        # update V
        if (ref < 0):
            V = V - 1 / tau_V * (V - V_L) * dt + I * R / tau_V * dt
        V_vec[i] = V
        
        # boundary conditions
        if (V >= V_th):
            spktimes.append(t)
            V = V_r
            ref = tau_ref
        ref = max(-1, ref - dt)

    return {
        't': t_vec,
        'I': I_vec,
        'V': V_vec,
        'spike_times': spktimes
    }


def sim_network():
    
    ## parameters
    N = 50

    j = 0.8
    sigma_J = 0.01
    p = 0.2

    ext = 0.2
    V_thr = 4.6
    V_reset = 0
    tau_m = 20
    tau_ref = 5
    tau_syn = 4

    dt = 0.1
    t_total = 2000

    ## construct network
    J = (j * (np.ones([N, N]) + sigma_J * np.random.normal(size=[N, N]))) * (np.random.random(size=[N, N]) < p).astype(float)
    J -= np.diag(np.diag(J))

    # initialization
    t_vec = np.arange(0, t_total, dt)
    fired = np.array([]) 
    binfired = []    
    V = 10 * np.random.random(size=N) # initial membrane potential
    Vth = V_thr * np.ones(N)
    tauV = tau_m * np.ones(N)
    refr = np.zeros(N)
    Isyn = np.zeros(N)
    Iext = ext * np.ones(N)

    for t in t_vec:

        # reset neurons that just spiked
        V[V >= Vth] = V_reset

        # update synaptic current
        Isyn -= Isyn / tau_syn * dt
        if np.any(fired):
            Isyn += np.sum(J[:, fired], 1) / tau_syn
        
        # update stimulus current
        if (t >= 300) and (t <= 500):
            Istim = 1
        elif (t >= 1400) and (t <= 1600):
            Istim = -1
        else:
            Istim = 0
            
        # update membrane potential    
        V = V - V / tauV * dt + (Isyn + Iext + Istim) * dt

        # refractory period
        V[refr > 0] = V_reset
        refr = np.maximum(-1, refr - dt)

        # boundary conditions
        fired = np.flatnonzero(V >= Vth)
        if len(fired) != 0: 
            V[fired] = V_reset 
            refr[fired] = tau_ref 
            binfired.append(np.hstack([t * np.ones([len(fired), 1]), fired.reshape([-1, 1])]))
                
    return np.vstack(binfired)


def sim_DDM(options):

    # extract
    nu = options['nu'] / 1000
    s = options['s'] / 400
    sigma = options['sigma']
    y_0 = options['y_0']
    dt = options['dt']
    t_total = options['t_total']
    upper_bound = options['upper_bound']
    lower_bound = options['lower_bound']

    # initialize
    t_vec = np.array([i * dt for i in range(int(t_total / dt))])
    y_vec = np.full(len(t_vec), np.nan)
    y = y_0
    y_vec[0] = y

    # loop over time
    decision_point = None
    for i in range(1, len(t_vec)):
        dy = nu * dt + np.sqrt(dt) * s * np.random.normal(loc=0, scale=sigma)
        y += dy
        y_vec[i] = y
        if (y >= upper_bound) or (y <= lower_bound):
            decision_point = [t_vec[i], y]
            break

    return {
        't': t_vec,
        'y': y_vec,
        'decision_point': decision_point
    }