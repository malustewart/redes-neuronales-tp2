import numpy as np

# Default parameters
default_k = 1.0  # h+n = k in approx 2
default_V0 = -80
default_I = 1
default_I_min = -100
default_I_max = 300
default_tmax = 300

g_Na = 120
g_K = 36
g_L = 0.3
V_Na = 50
V_K = -77
V_L = -54.4
C = 1e-6

class Conductances:
    def __init__(self, g_Na=g_Na, g_K=g_K, g_L=g_L):
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L

class MhnParameters:
    def __init__(self, V):
        a_m = 0.1 * (V + 40) / (1 - np.exp((-V - 40) / 10))
        b_m = 4 * np.exp((-V - 65) / 18)
        a_h = 0.07 * np.exp((-V - 65) / 20)
        b_h = 1 / (1 + np.exp((-V - 35) / 10))
        a_n = 0.01 * (V + 55) / (1 - np.exp((-V - 55) / 10))
        b_n = 0.125 * np.exp((-V - 65) / 80)

        m_tau = 1 / (a_m + b_m)
        h_tau = 1 / (a_h + b_h)
        n_tau = 1 / (a_n + b_n)

        m_inf = m_tau * a_m
        h_inf = h_tau * a_h
        n_inf = n_tau * a_n

        self.m_inf = m_inf
        self.h_inf = h_inf
        self.n_inf = n_inf
        self.m_tau = m_tau
        self.h_tau = h_tau
        self.n_tau = n_tau

class InversionV:
    def __init__(self, V_Na=V_Na, V_K=V_K, V_L=V_L):
        self.V_Na = V_Na
        self.V_K = V_K
        self.V_L = V_L

class HHParams:
    def __init__(self, conductances=None, inversion_v=None, I=default_I, C=C, k=default_k, Vsyn1=0, Vsyn2=0, tau = 3, gsyn1=1, gsyn2=1):
        self.conductances = conductances if conductances else Conductances()
        self.inversion_v = inversion_v if inversion_v else InversionV()
        self.I = I
        self.C = C
        self.k = k
        self.Vsyn1  = Vsyn1
        self.Vsyn2  = Vsyn2
        self.tau = tau
        self.gsyn1 = gsyn1
        self.gsyn2 = gsyn2

def S(V):
    return 0.5 * (1 + np.tanh(V / 5))

def HH_standard(yk, params: HHParams):
    V1, m1, h1, n1, s1, V2, m2, h2, n2, s2 = yk
    V_Na = params.inversion_v.V_Na
    V_K = params.inversion_v.V_K
    V_L = params.inversion_v.V_L
    g_Na = params.conductances.g_Na
    g_K = params.conductances.g_K
    g_L = params.conductances.g_L
    I = params.I
    tau = params.tau
    mhn1 = MhnParameters(V1)
    mhn2 = MhnParameters(V2)
    Vsyn1 = params.Vsyn1
    Vsyn2 = params.Vsyn2
    gsyn1 = params.gsyn1
    gsyn2 = params.gsyn2

    Iion1 = g_Na * m1**3 * h1 * (V1 - V_Na) + g_K * n1**4 * (V1 - V_K) + g_L * (V1 - V_L)
    Iion2 = g_Na * m2**3 * h2 * (V2 - V_Na) + g_K * n2**4 * (V2 - V_K) + g_L * (V2 - V_L)

    Isyn1 = gsyn1 * s1 * (V1 - Vsyn1)
    Isyn2 = gsyn2 * s2 * (V2 - Vsyn2)

    dyk = np.zeros(10)
    dyk[0] = (I - Isyn1 - Iion1) / params.C
    dyk[1] = (mhn1.m_inf - m1) / mhn1.m_tau
    dyk[2] = (mhn1.h_inf - h1) / mhn1.h_tau
    dyk[3] = (mhn1.n_inf - n1) / mhn1.n_tau
    dyk[4] = (S(V2) - s1) / tau
    dyk[5] = (I - Isyn2 - Iion2) / params.C
    dyk[6] = (mhn2.m_inf - m2) / mhn2.m_tau
    dyk[7] = (mhn2.h_inf - h2) / mhn2.h_tau
    dyk[8] = (mhn2.n_inf - n2) / mhn2.n_tau
    dyk[9] = (S(V1) - s2) / tau

    return dyk