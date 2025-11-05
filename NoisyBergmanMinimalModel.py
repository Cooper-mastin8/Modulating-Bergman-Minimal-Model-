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
p1, p2, p3, n, gamma, Gb, Ib = [0.028, 0.025, 5.0e-5, 0.2, 0.0005, 90.0, 10.0]
D_G = 75.0      # Glucose infusion rate [mg/dL per min]
D_I = 25.0      # Insulin infusion rate for first phase [μU/ml per min]
params = [p1, p2, p3, n, gamma, Gb, Ib, D_G, D_I]

# --- INITIAL CONDITIONS ---
t_eval = np.linspace(0, 180, 1000)
initial_conditions = [90.0, 0.0, 10.0]

# --- SOLVE THE DETERMINISTIC MODEL ---
solution = solve_ivp(
    lambda t, y: bergman_model(t, y, params),
    [0, 180], 
    initial_conditions,
    dense_output=True,
    t_eval=t_eval
)

# Extract the clean, deterministic results
t = solution.t
glucose_clean = solution.y[0]
insulin_action = solution.y[1] 
insulin_clean = solution.y[2]

# --- ADD RANDOM NOISE TO SIMULATE MEASUREMENT ERROR ---
# Define the standard deviation of the noise for each signal
glucose_noise_std = 2.0  # Noise with a std dev of 3.0 mg/dL
insulin_noise_std = 1.5  # Noise with a std dev of 2.0 μU/ml

# Generate and add Gaussian noise
glucose_noisy = glucose_clean + np.random.randn(len(t)) * glucose_noise_std
insulin_noisy = insulin_clean + np.random.randn(len(t)) * insulin_noise_std


# --- PLOT AND COMPARE ---
plt.figure(figsize=(15, 4))

# Plot Glucose (Clean vs. Noisy)
plt.subplot(1, 3, 1)
plt.plot(t, glucose_clean, 'k--', linewidth=2, label='Deterministic')
plt.plot(t, glucose_noisy, 'b-', linewidth=1.5, label='Noisy')
plt.title('Glucose Concentration')
plt.xlabel('Time (minutes)')
plt.ylabel('Glucose (mg/dl)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Insulin (Clean vs. Noisy)
plt.subplot(1, 3, 2) 
plt.plot(t, insulin_clean, 'k--', linewidth=2, label='Deterministic')
plt.plot(t, insulin_noisy, 'r-', linewidth=1.5, label='Noisy')
plt.title('Insulin Concentration')
plt.xlabel('Time (minutes)')
plt.ylabel('Insulin (μU/ml)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Insulin Action (unchanged)
plt.subplot(1, 3, 3)
plt.plot(t, insulin_action, 'g-', linewidth=2)
plt.title('Remote Insulin Action')
plt.xlabel('Time (minutes)')
plt.ylabel('Remote Insulin Action')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print the peak values from the noisy signals
print(f"Peak noisy glucose: {max(glucose_noisy):.1f} mg/dl")
print(f"Peak noisy insulin: {max(insulin_noisy):.1f} μU/ml")