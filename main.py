import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import HH
import sys
import os
from scipy.signal import find_peaks

def plot_HH_time(sol, save_path=None, label=""):
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 1, 1)
    plt.plot(sol.t, sol.y[0], label='V1 (mV)', color='tab:blue')
    plt.plot(sol.t, sol.y[5], label='V2 (mV)', color='tab:orange')
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    plt.title('Membrane Potentials (V1, V2)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(sol.t, sol.y[4], label='s1', color='tab:green')
    plt.plot(sol.t, sol.y[9], label='s2', color='tab:red')
    plt.xlabel('Time (ms)')
    plt.ylabel('s')
    plt.title('Synaptic Variables (s1, s2)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{label}_time.png"))
        plt.close()
    else:
        plt.show()

def plot_phase_diagram(sol, N=1000, save_path=None, label=""):
    plt.figure(figsize=(6, 6))
    plt.plot(sol.y[0][-N:], sol.y[5][-N:], color='tab:purple')
    plt.xlabel('V1 (mV)')
    plt.ylabel('V2 (mV)')
    plt.title(f'Phase Diagram: V1 vs V2 (last {N} points)')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{label}_phase.png"))
        plt.close()
    else:
        plt.show()

def default_y0(V10=-70, V20=10):
    mhn01 = HH.MhnParameters(V10)
    mhn02 = HH.MhnParameters(V20)
    return [
        V10, mhn01.m_inf, mhn01.h_inf, mhn01.n_inf, HH.S(V10),
        V20, mhn02.m_inf, mhn02.h_inf, mhn02.n_inf, HH.S(V20)
    ]

def get_spike_times(V, t):
    """
    Detects spike times as local maxima in V using scipy.signal.find_peaks.
    Returns the times at which peaks occur.
    kwargs are passed to find_peaks (e.g., height, distance).

    Assumes neuron is in stationary regime.
    """
    Vmax = np.max(V)
    Vmin = np.min(V)
    V_range = Vmax - Vmin
    threshold = Vmax - 0.3 * V_range
    peaks, _ = find_peaks(V,height=threshold)
    return t[peaks]

def spike_offset(sol):
    """
    Calculates the offset between spikes of V1 and V2, normalized by the spike period T.
    Returns the mean offset (as a fraction of T) and the period T.
    """
    mid = len(sol.t) // 2  # Analyze second half to ensure stationarity
    V1 = sol.y[0][mid:]
    V2 = sol.y[5][mid:]
    t = sol.t[mid:]

    spikes1 = get_spike_times(V1, t)
    spikes2 = get_spike_times(V2, t)

    min_spikes = min(len(spikes1), len(spikes2))
    if min_spikes < 2:
        return None, None  # Not enough spikes to calculate offset

    # Calculate period T from V1 spikes
    T = np.mean(np.diff(spikes1))

    # For each spike in V1, find the nearest spike in V2 and compute offset
    offsets = []
    for t1 in spikes1:
        # Find the closest spike in V2 after t1
        after = spikes2[spikes2 >= t1]
        before = spikes2[spikes2 < t1]
        if len(after) > 0 and len(before) > 0:
            # Take the closest (either before or after)
            t2 = after[0] if (after[0] - t1) < (t1 - before[-1]) else before[-1]
        elif len(after) > 0:
            t2 = after[0]
        elif len(before) > 0:
            t2 = before[-1]
        else:
            continue
        offsets.append(((t2 - t1) % T) / T)  # Normalize to [0,1)

    if len(offsets) == 0:
        return None, None

    mean_offset = np.mean(offsets)
    return mean_offset, T

def run_case(case, save_path=None):
    params = HH.HHParams(**case.get('params', {}))
    y0 = case.get('y0', default_y0())
    t_span = case.get('t_span', (0, HH.default_tmax))
    def hh_ode(t, y):
        return HH.HH_standard(y, params)
    return solve_ivp(hh_ode, t_span, y0, method='Radau')

if __name__ == "__main__":
    save_path = None
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
        os.makedirs(save_path, exist_ok=True)

    cases = [
        {
            'label': 'Default',
        },
        {
            'label': 'Low Coupling',
            'params': {'gsyn1': 0.2, 'gsyn2': 0.2},
            'y0': default_y0(V10=-60, V20=0)
        },
        {
            'label': 'High Current',
            'params': {'I': 50}
        },
        {
            'label': 'Bokita',
            'params': {'I': 10, 'gsyn1': 1, 'gsyn2': 1, 'Vsyn1': 0, 'Vsyn2': 0},
            'y0': default_y0(V10=-70, V20=-20)
        },
        {
            'label': 'XXXXXXXXXXXXXXX',
            'params': {'I': 10, 'gsyn1': 1, 'gsyn2': 1, 'Vsyn1': 0, 'Vsyn2': 0},
            'y0': default_y0(V10=-70, V20=20)
        },
    ]

    for case in cases:
        sol = run_case(case, save_path=save_path)
        mean_offset, period = spike_offset(sol)
        if mean_offset is not None:
            print(f"{case.get('label', 'Unnamed')}: Mean spike offset = {mean_offset:.3f} (fraction of period), Period T = {period:.3f} ms")
        else:
            print(f"{case.get('label', 'Unnamed')}: Not enough spikes to calculate offset.")
        label = case.get('label', 'Unnamed')
        plot_HH_time(sol, save_path=save_path, label=label)
        plot_phase_diagram(sol, save_path=save_path, label=label)

