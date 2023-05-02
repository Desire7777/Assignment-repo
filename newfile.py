import numpy as np

import matplotlib.pyplot as plt


class PotentialEnergy:
    def __init__(self, a, b, V):
        self.a = a
        self.b = b
        self.V = V
    
    def get_energy(self, x):
        if x < 0 or x > self.b:
            return float('inf')
        elif x >= self.a:
            return self.V
        else:
            return 0

class WaveFunction:
    def __init__(self, a, b, N):
        self.a = a
        self.b = b
        self.N = N
        self.dx = (b - a) / (N - 1)
        self.x = np.linspace(a, b, N)
        self.psi = np.zeros(N)
    
    def set_initial_condition(self, sigma):
        self.psi = np.exp(-0.5*((self.x-self.a)/sigma)**2)
    
    def set_boundary_condition(self, psi_0, psi_N):
        self.psi[0] = psi_0
        self.psi[-1] = psi_N
    
    def get_wave_function(self):
        return self.x, self.psi
    
    def evolve(self, dt, potential_energy):
        hbar = 1
        m = 1
        alpha = 1j*hbar*dt/(2*m*self.dx**2)
        beta = 1 + 2*alpha
        gamma = -alpha
        
        V = np.array([potential_energy.get_energy(x) for x in self.x])
        
        for i in range(1, self.N-1):
            self.psi[i] = alpha*self.psi[i-1] + beta*self.psi[i] + gamma*self.psi[i+1] - alpha*dt*V[i]*self.psi[i]


class EigenvalueProblem:
    def __init__(self, a, b, V, N):
        self.a = a
        self.b = b
        self.V = V
        self.N = N
        
        self.potential_energy = PotentialEnergy(a, b, V)
        
        self.wave_function = WaveFunction(a, b, N)
        self.wave_function.set_initial_condition(sigma=(b-a)/10)
        self.wave_function.set_boundary_condition(psi_0=0, psi_N=0)
        
        self.eigenvalues = []
        self.eigenvectors = []
    
    def solve(self, max_iterations=100, tolerance=1e-6):
        for i in range(max_iterations):




# Define the grid of values for a and V{}
a_values = np.linspace(0.1, 0.9, 20)
V_values = np.linspace(1, 10, 20)

# Define the maximum number of eigenvalues to compute
max_eigenvalues = 10

# Initialize the heat maps
energy_heatmap = np.zeros((len(a_values), len(V_values)))
wavefunction_heatmap = np.zeros((len(a_values), len(V_values), max_eigenvalues, len(x)))

# Loop over the values of a and V{}
for i, a in enumerate(a_values):
    for j, V in enumerate(V_values):
        eigenvalue_problem = EigenvalueProblem(a=a, b=1, V=V, N=1000)
        eigenvalue_problem.solve()
        
        # Compute the energy eigenvalues and wave functions
        eigenvalues = eigenvalue_problem.eigenvalues[:max_eigenvalues]
        wavefunctions = eigenvalue_problem.eigenvectors[:max_eigenvalues]
        
        # Store the energy eigenvalues in the heat map
        energy_heatmap[i, j] = eigenvalues[0]
        
        # Store the wave functions in the heat map
        for k in range(len(eigenvalues)):
            wavefunction_heatmap[i, j, k] = np.abs(wavefunctions[k])**2

# Plot the heat maps
fig, axs = plt.subplots(2, 1, figsize=(8, 12))
axs[0].imshow(energy_heatmap, origin='lower', extent=[V_values[0], V_values[-1], a_values[0], a_values[-1]], aspect='auto')
axs[0].set_xlabel('V{}')
axs[0].set_ylabel('a')
axs[0].set_title('Energy Eigenvalues')
axs[1].imshow(wavefunction_heatmap[:,:,0,:], origin='lower', extent=[V_values[0], V_values[-1], a_values[0], a_values[-1]], aspect='auto')
axs[1].set_xlabel('V{}')
axs[1].set_ylabel('a')
axs[1].set_title('Wave Functions')
plt.show()
