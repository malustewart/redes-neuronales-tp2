import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import HH
import WC
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
        plt.savefig(os.path.join(save_path, f"{label}_phase.pdf"))
        plt.close()
    else:
        plt.show()

def plot_offset_time(offsets, t_offsets, label, sublabels, save_path=None):
    plt.figure(figsize=(10, 4))
    [plt.plot(t_offset, offset, marker='o', linestyle='-', label=sublabel) for offset, t_offset, sublabel in zip(offsets, t_offsets, sublabels)]
    plt.xlabel('Time (ms)')
    plt.ylim(0, np.max(np.hstack(offsets))*1.1)
    plt.ylabel('Spike Offset (radians)')
    plt.title('Spike Offset Over Time - ' + label)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{label}_offset_time.png"))
        plt.close()
    else:
        plt.show()

def plot_f_vs_gsyn(f_values, gsyn_values, save_path=None, label=""):
    plt.figure(figsize=(8, 5))
    plt.plot(gsyn_values, f_values, marker='o')
    plt.xlabel('Synaptic Conductance (gsyn)')
    plt.ylabel('Frequency (kHz)')
    plt.title('Frequency vs Synaptic Conductance - ' + label)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{label}_f_vs_gsyn.png"))
        plt.close()
    else:
        plt.show()

def plot_WC_time(sol, save_path=None, label=""):
    plt.figure(figsize=(10, 7))
    plt.plot(sol.t, sol.y[0], label='Excitatory', color='tab:blue')
    plt.plot(sol.t, sol.y[1], label='Inhibitory', color='tab:orange')
    plt.xlabel('Time (ms)')
    plt.ylabel('Population activity')
    plt.title('Excitatory and inhibitory population activity in time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        print(save_path, f"{label}_time.png")
        plt.savefig(os.path.join(save_path, f"{label}_time.png"))
        plt.close()
    else:
        plt.show()

def default_y0_HH(V10=-70, V20=10):
    mhn01 = HH.MhnParameters(V10)
    mhn02 = HH.MhnParameters(V20)
    return [
        V10, mhn01.m_inf, mhn01.h_inf, mhn01.n_inf, HH.S(V10),
        V20, mhn02.m_inf, mhn02.h_inf, mhn02.n_inf, HH.S(V20)
    ]

def default_y0_WC():
    he = 1
    hi = 1
    return [he, hi]

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

def last_period(sol):
    V1 = sol.y[0][:]
    t = sol.t[:]
    spikes = get_spike_times(V1, t)
    if len(spikes) < 2:
        return None  # Not enough spikes to determine period
    return spikes[-1] - spikes[-2]

def spike_offset(sol):
    V1 = sol.y[0][:]
    V2 = sol.y[5][:]
    t = sol.t[:]

    spikes1 = get_spike_times(V1, t)
    spikes2 = get_spike_times(V2, t)

    min_spikes = min(len(spikes1), len(spikes2))
    if min_spikes < 2:
        return None, None  # Not enough spikes to calculate offset

    # Calculate period T from V1 spikes
    Ts1 = np.diff(spikes1)

    # For each spike in V1, find the nearest spike in V2 and compute offset
    offsets = []
    t_offset = []
    for T, t1 in zip(Ts1, spikes1):
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
        offsets.append(2*np.pi*(abs(t2 - t1) % T) / T)
        t_offset.append(t1)

    if not offsets:
        return None, None

    return offsets[1:], t_offset[1:]

def run_case_HH(case):
    params = HH.HHParams(**case.get('params', {}))
    y0 = case.get('y0', default_y0_HH())
    t_span = case.get('t_span', (0, HH.default_tmax))
    def hh_ode(t, y):
        return HH.HH_standard(y, params)
    return solve_ivp(hh_ode, t_span, y0, method='Radau')

def run_case_WC(case):
    params = WC.WCParams(**case.get('params', {}))
    y0 = case.get('y0', default_y0_WC())
    t_span = case.get('t_span', (0, WC.default_tmax))
    def wc_ode(t, y):
        return WC.WC(y, params)
    return solve_ivp(wc_ode, t_span, y0, method='Radau')

def punto_1_A_1(save_path: str):
    Vsyn = 0
    I = 10
    cases_1_A_1 = [
        {
            'label': f'1A1 gsyn={gsyn}',
            'params': {'I': I, 'gsyn1': gsyn, 'gsyn2': gsyn, 'Vsyn1': Vsyn, 'Vsyn2': Vsyn},
            'y0': default_y0_HH(V10=-70, V20=V20),
            't_span': (0, 500)
        }
        for gsyn, V20 in zip([0.05, 0.1, 0.5], [0, -15, -54])
    ]

    offsets = []
    t_offsets = []
    sublabels = []
    print("1 A 1 running")
    for case in cases_1_A_1:
        label = f"{case.get('label', '')}"
        print(f"Running case: {label}")
        sol = run_case_HH(case)
        offset, t_offset = spike_offset(sol)
        if offset:
            offsets.append(offset)
            t_offsets.append(t_offset)
            sublabels.append(case.get('label', ''))
        # plot_HH_time(sol, save_path=save_path, label=label)
    print("1 A 1 plotting")
    plot_offset_time(offsets, t_offsets, save_path=save_path, label=f"Vsyn={Vsyn}mV - I={I}uA", sublabels=sublabels)

def punto_1_B_1(save_path: str):
    I_1_B_1 = 10
    Vsyn_1_B_1 = 0
    cases_1_B_1 = [
        {
            'label': f'gsyn={gsyn}',
            'params': {'I': I_1_B_1, 'gsyn1': gsyn, 'gsyn2': gsyn, 'Vsyn1': Vsyn_1_B_1, 'Vsyn2': Vsyn_1_B_1},
            'y0': default_y0_HH(V10=-70, V20=-70),
            't_span': (0, 80)
        }
        for gsyn in np.linspace(0, 1, 6)
    ]

    f_1_B_1 = []
    gsyns_1_B_1 = []
    sublabels_1_B_1 = []
    for case in cases_1_B_1:
        print(f"Running case: {case.get('label', '')}")
        sol = run_case_HH(case)
        T = last_period(sol)
        if T:
            f_1_B_1.append(1/T)
            gsyns_1_B_1.append(case["params"]["gsyn1"])
        # plot_HH_time(sol, save_path=save_path, label=f"Vsyn={Vsyn_1_B_1}mV - I={I_1_B_1}uA "+case.get('label', '')) # debug
    plot_f_vs_gsyn(f_1_B_1, gsyns_1_B_1, save_path=save_path, label=f"Vsyn={Vsyn_1_B_1}mV - I={I_1_B_1}uA")

def punto_1_C_1(save_path: str):
    Vsyn_1_C_1 = -80
    I_1_C_1 = 10
    cases_1_C_1 = [
        {
            'label': f'1C1 gsyn={gsyn}',
            'params': {'I': I_1_C_1, 'gsyn1': gsyn, 'gsyn2': gsyn, 'Vsyn1': Vsyn_1_C_1, 'Vsyn2': Vsyn_1_C_1},
            'y0': default_y0_HH(V10=-70, V20=-70),
            't_span': (0, 500)
        }
        for gsyn in [0.05, 0.1, 0.5, 1, 2]
    ]

    offsets_1_C_1 = []
    t_offsets_1_C_1 = []
    sublabels_1_C_1 = []
    print("1 C 1 running")
    for case in cases_1_C_1:
        label = f"{case.get('label', '')}"
        print(f"Running case: {label}")
        sol = run_case_HH(case)
        offset, t_offset = spike_offset(sol)
        if offset:
            offsets_1_C_1.append(offset)
            t_offsets_1_C_1.append(t_offset)
            sublabels_1_C_1.append(case.get('label', ''))
        plot_HH_time(sol, save_path=save_path, label=label)
    print("1 C 1 plotting")
    plot_offset_time(offsets_1_C_1, t_offsets_1_C_1, save_path=save_path, label=f"Vsyn={Vsyn_1_C_1}mV - I={I_1_C_1}uA", sublabels=sublabels_1_C_1)

def punto_1_C_2(save_path: str):
    print("1 C 2 running")
    I_1_C_2 = 10
    Vsyn_1_C_2 = -80
    cases_1_C_2 = [
        {
            'label': f'gsyn={gsyn}',
            'params': {'I': I_1_C_2, 'gsyn1': gsyn, 'gsyn2': gsyn, 'Vsyn1': Vsyn_1_C_2, 'Vsyn2': Vsyn_1_C_2},
            'y0': default_y0_HH(V10=-74, V20=40),
            't_span': (0, 400)
        }
        for gsyn in np.linspace(0, 1, 6)
    ]

    f_1_C_2 = []
    gsyns_1_C_2 = []
    for case in cases_1_C_2:
        print(f"Running case: {case.get('label', '')}")
        sol = run_case_HH(case)
        T = last_period(sol)
        if T:
            f_1_C_2.append(1/T)
            gsyns_1_C_2.append(case["params"]["gsyn1"])
        plot_HH_time(sol, save_path=save_path, label=f"Vsyn={Vsyn_1_C_2}mV - I={I_1_C_2}uA "+case.get('label', '')) # debug
    plot_f_vs_gsyn(f_1_C_2, gsyns_1_C_2, save_path=save_path, label=f"Vsyn={Vsyn_1_C_2}mV - I={I_1_C_2}uA")

def punto_2_A_1(save_path: str):
    # tau = 1
    # g_ee = 1
    # g_ei = 1
    # g_ie = 1
    # g_ii = 1
    # I_e = 10
    # I_i = 10

    cases = [
        {
            'label': f'2A1 I_e=I_i={I} he0={he0} hi0={hi0}',
            'params': {'I_e': I, 'I_i': I},
            'y0': [hi0, he0],
            't_span': (0, 100)
        }
        for I in [1,-1]
        for hi0,he0 in zip([-0.1,-5,-10],[-0.2, -7, -11])
    ]

    print("2 A 1 running")
    for case in cases:
        label = f"{case.get('label', '')}"
        print(f"Running case: {label}")
        sol = run_case_WC(case)
        plot_WC_time(sol, save_path=save_path, label=label)

def punto_2_A_2(save_path: str):
    # tau = 1
    # g_ee = 1
    g_ei = 2
    # g_ie = 1
    g_ii = 1
    # I_e = 10
    # I_i = 10

    cases = [
        {
            'label': f'2A2 I_e={I_e} I_i={I_i} he0={he0} hi0={hi0}',
            'params': {'I_e': I_e, 'I_i': I_i, 'g_ei': g_ei, 'g_ii': g_ii},
            'y0': [hi0, he0],
            't_span': (0, 100)
        }
        for I_e,I_i in zip([1], [2])
        for hi0,he0 in zip([-5,-5],[-0.2, 2])
    ]

    print("2 A 2 running")
    for case in cases:
        label = f"{case.get('label', '')}"
        print(f"Running case: {label}")
        sol = run_case_WC(case)
        plot_WC_time(sol, save_path=save_path, label=label)

def punto_2_A_4(save_path: str):
    # tau = 1
    # g_ee = 1
    # g_ei = 1
    # g_ie = 1
    # g_ii = 1
    # I_e = 10
    # I_i = 10

    cases = [
        {
            'label': f'2A4 I_e={I_e} I_i={I_i} he0={he0} hi0={hi0}',
            'params': {'I_e': I_e, 'I_i': I_i},
            'y0': [hi0, he0],
            't_span': (0, 100)
        }
        for I_e,I_i in zip([1], [1])
        for hi0,he0 in zip([5,3],[1, 2])
    ]

    print("2 A 4 running")
    for case in cases:
        label = f"{case.get('label', '')}"
        print(f"Running case: {label}")
        sol = run_case_WC(case)
        plot_WC_time(sol, save_path=save_path, label=label)


if __name__ == "__main__":
    save_path = None
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
        os.makedirs(save_path, exist_ok=True)

    # punto_1_A_1(save_path)
    # punto_1_B_1(save_path)
    # punto_1_C_1(save_path)
    # punto_1_C_2(save_path)
    # punto_2_A_1(save_path)
    # punto_2_A_2(save_path)
    punto_2_A_4(save_path)





