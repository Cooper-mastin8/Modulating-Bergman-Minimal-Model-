import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def bergman_model(t, y, params):
    G, X, I = y
    p1, p2, p3, n, gamma, Gb, Ib, D_G, D_I = params
    
    # --- MODEL THE BOLUS AS TWO SIMULTANEOUS INFUSIONS ---
    # 1. Glucose infusion for the first 2 minutes
    glucose_infusion = D_G if t < 2.0 else 0.0
    
    # 2. First-phase insulin infusion for the first 5 minutes
    insulin_infusion = D_I if t < 5.0 else 0.0

    # --- EQUATIONS ---
    # Glucose equation includes its infusion
    dG_dt = -(p1 + X) * G + p1 * Gb + glucose_infusion
    
    # Insulin equation now includes its own infusion, but no dG/dt term
    second_phase_release = gamma * max(0, G - Gb)
    dI_dt = -n * (I - Ib) + second_phase_release + insulin_infusion
    
    # Remote insulin action is unchanged
    dX_dt = -p2 * X + p3 * (I - Ib)
    
    return [dG_dt, dX_dt, dI_dt]

# --- PARAMETERS ---
# Canonical parameters for the core model
p1, p2, p3, n, gamma, Gb, Ib = [0.028, 0.025, 5.0e-5, 0.2, 0.0005, 90.0, 10.0]

# Infusion parameters to produce the rise
D_G = 75.0      # Glucose infusion rate [mg/dL per min]
D_I = 25.0      # Insulin infusion rate for first phase [μU/ml per min]

params = [p1, p2, p3, n, gamma, Gb, Ib, D_G, D_I]

# --- INITIAL CONDITIONS ---
# Start at the true basal state, before the injection
t = np.linspace(0, 120, 1000)
initial_conditions = [90.0, 0.0, 10.0]

# --- SOLVE AND PLOT ---
solution = solve_ivp(
    lambda t, y: bergman_model(t, y, params),
    [0, 180], 
    initial_conditions,
    t_eval=t
)

glucose = solution.y[0]
insulin_action = solution.y[1] 
insulin = solution.y[2]

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(t, glucose, 'b-', linewidth=2)
plt.title('Glucose Concentration')
plt.xlabel('Time (minutes)')
plt.ylabel('Glucose (mg/dl)')
plt.grid(True, alpha=0.3)
plt.subplot(1, 3, 2) 
plt.plot(t, insulin, 'r-', linewidth=2)
plt.title('Insulin Concentration')
plt.xlabel('Time (minutes)')
plt.ylabel('Insulin (μU/ml)')
plt.grid(True, alpha=0.3)
plt.subplot(1, 3, 3)
plt.plot(t, insulin_action, 'g-', linewidth=2)
plt.title('Remote Insulin Action')
plt.xlabel('Time (minutes)')
plt.ylabel('Remote Insulin Action')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Peak glucose: {max(glucose):.1f} mg/dl")
print(f"Peak insulin: {max(insulin):.1f} μU/ml")