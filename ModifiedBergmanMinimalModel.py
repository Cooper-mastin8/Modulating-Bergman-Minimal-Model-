import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def bergman_model_muscle(t, y, params):
    """
    Extended Bergman Minimal Model with muscle-mass–dependent insulin sensitivity (p3 term).
    """
    G, X, I = y
    (
        p1, p2, p30, n, gamma, Gb, Ib, D_G, D_I,
        M, M0, beta  # muscle-related parameters
    ) = params
    
    #Compute p3 as a function of muscle mass
    p3 = p30 * (1 + beta * (M - M0) / M0)
    
   #Glucose infusion for first 2 minutes and insulin infusion for first 5 minutes
    glucose_infusion = D_G if t < 2.0 else 0.0
    insulin_infusion = D_I if t < 5.0 else 0.0
    
    # --- Differential equations
    second_phase_release = gamma * max(0, G - Gb)
    dG_dt = -(p1 + X) * G + p1 * Gb + glucose_infusion
    dI_dt = -n * (I - Ib) + second_phase_release + insulin_infusion
    dX_dt = -p2 * X + p3 * (I - Ib)
    
    return [dG_dt, dX_dt, dI_dt]

# --- PARAMETERS (set all to None for now) ---
p1 = 0.028      # glucose effectiveness
p2 = 0.025      # insulin action decay rate
p30 = 5.0e-5    # baseline p3 value (will be fitted or chosen)
n = 0.2         # insulin clearance rate
gamma = 0.0005  # 2nd-phase insulin release sensitivity
Gb = 90.0       # basal glucose (mg/dL)
Ib = 10.0       # basal insulin (μU/mL)
D_G = 75.0      # glucose bolus amplitude
D_I = 25.0      # insulin bolus amplitude
M = 160        # muscle mass of subject
M0 = 147      # reference muscle mass (from dataset)
beta = 0.5     # slope relating p3 to muscle mass

params = [p1, p2, p30, n, gamma, Gb, Ib, D_G, D_I, M, M0, beta]

# --- SIMULATION SETUP ---
t = np.linspace(0, 180, 1000)
initial_conditions = [Gb, 0.0, Ib]  # start at basal state (to be filled in later)

# Replace single-run simulation + plotting with a sweep over M values
M_values = np.arange(100, 201, 20)  # 100,120,...,200
colors = plt.cm.viridis(np.linspace(0, 1, len(M_values)))

plt.figure(figsize=(15, 10))

# Subplot 1: Glucose
plt.subplot(3, 1, 1)
for i, M_val in enumerate(M_values):
    params_mod = params.copy()
    params_mod[9] = M_val  # index 9 is M in params list
    sol = solve_ivp(lambda t, y: bergman_model_muscle(t, y, params_mod),
                    [0, 180], initial_conditions, t_eval=t)
    glucose, X, insulin = sol.y
    plt.plot(t, glucose, color=colors[i], label=f'M={M_val}')
plt.title('Glucose Concentration for different M')
plt.xlabel('Time (min)'); plt.ylabel('Glucose (mg/dL)')
plt.legend(loc='upper right'); plt.grid(True)

# Subplot 2: Insulin
plt.subplot(3, 1, 2)
for i, M_val in enumerate(M_values):
    params_mod = params.copy()
    params_mod[9] = M_val
    sol = solve_ivp(lambda t, y: bergman_model_muscle(t, y, params_mod),
                    [0, 180], initial_conditions, t_eval=t)
    glucose, X, insulin = sol.y
    plt.plot(t, insulin, color=colors[i], label=f'M={M_val}')
plt.title('Insulin Concentration for different M')
plt.xlabel('Time (min)'); plt.ylabel('Insulin (μU/mL)')
plt.legend(loc='upper right'); plt.grid(True)

# Subplot 3: Remote Insulin Action X
plt.subplot(3, 1, 3)
for i, M_val in enumerate(M_values):
    params_mod = params.copy()
    params_mod[9] = M_val
    sol = solve_ivp(lambda t, y: bergman_model_muscle(t, y, params_mod),
                    [0, 180], initial_conditions, t_eval=t)
    glucose, X, insulin = sol.y
    plt.plot(t, X, color=colors[i], label=f'M={M_val}')
plt.title('Remote Insulin Action for different M')
plt.xlabel('Time (min)'); plt.ylabel('X (a.u.)')
plt.legend(loc='upper right'); plt.grid(True)

plt.tight_layout(); plt.show()
