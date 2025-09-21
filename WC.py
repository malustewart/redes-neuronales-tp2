import numpy as np
from dataclasses import dataclass

default_tmax = 300

@dataclass
class WCParams:
    tau: float = 1.0
    g_ee: float = 1.0
    g_ei: float = 1.0
    g_ie: float = 1.0
    g_ii: float = 1.0
    I_e: float = 0.0
    I_i: float = 0.0

def S(x):
    return np.heaviside(x,1)*x

def WC(y, params: WCParams):
    hi, he = y
    dyk = np.zeros(2)
    dyk[0] = (-he + params.g_ee*S(he) - params.g_ei*S(hi) + params.I_e)/params.tau
    dyk[1] = (-hi + params.g_ie*S(he) - params.g_ii*S(hi) + params.I_i)/params.tau
    return dyk