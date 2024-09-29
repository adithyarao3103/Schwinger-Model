# Numerical simulation of $U(1)$ Wilson Lattice Gauge Model - The Schwinger Model

In this repository, we present the numerical simulations of the Schwinger model, which is a $U(1)$ Wilson lattice gauge theory, by studying the dynamics of the spin-lattice model obtained by the Jordan-Wigner Transformation. 

We present both [real-time evolution](#real-time-dynamics) of particle density, entanglement entropy and electric fields for the vacuum of the Schwinger Model, and [variational quantum simulation](#variational-quantum-simulations) to obtain the ground state of the model for different mass parameters and observe the phase transitions in the model.  

## Real-Time Dynamics

[Jupyter Notebook](num_sim_schwinger.ipynb)

The numerical simulations of the real time dynamics is performed by both exact exponentiation of the Hamiltonian and Trotterization of the Hamiltonian. We present both the results below.

The theory and the experimental results are presented in the paper by Muschik et al.: https://iopscience.iop.org/article/10.1088/1367-2630/aa89ab

The results are as follows, and it can see that it matches the results in the paper.

### Particle Density and Half Chain Entanglement Entropy

![nu_and_s](nu_and_s.png)

### Electric Field

![elec_field](elec_field.png)

## Variational Quantum Simulations

[Jupyter Notebook](vqe_gs_schwinger.ipynb)

Here, we variationally obtain the ground state of the Schwinger Model and calculate order parameters to observe phase transition as a function of bare mass of fermions of the theory. 

The experimental protocol and results are presented in the paper by Kokail et al.: https://www.nature.com/articles/s41586-019-1177-4

> Please write to me if you find any error in the following analysis

To obtain the ground state wavefunction, we start with a random state as the initial state and we choose to rotate the state at each site by three angles $\theta_x$ $(R_x)$, $\theta_y$ $(R_y)$, & $\theta_z$ $(R_z)$

$$
R(\vec{\theta})= \left( \exp\left(\frac{i}{2} \sigma_x \theta_{x1} \right)\times \exp\left(\frac{i}{2} \sigma_y \theta_{y1} \right)\times \exp\left(\frac{i}{2} \sigma_z \theta_{z1} \right) \right) \otimes \cdots\otimes \left( \exp\left(\frac{i}{2} \sigma_x \theta_{xN} \right)\times \exp\left(\frac{i}{2} \sigma_y \theta_{yN} \right)\times \exp\left(\frac{i}{2} \sigma_z \theta_{zN} \right) \right)
$$

$$
\implies R(\vec\theta) = R_1(\vec\theta_1)\otimes R_2(\vec\theta_2) \otimes \cdots\otimes R_N(\vec\theta_N)
$$

where $R_i$ denotes rotation matrix acting on site $i$, giving a parameterized state 

$$
|\psi(\vec\theta)\rangle = R(\vec\theta)|\psi_{\text{init}}\rangle
$$

In this case, the loss function which we want to minimize is the energy itself, therefore

$$
\mathrm{Loss} = L= \langle \psi(\vec\theta)| H | \psi(\vec\theta)\rangle = L= \langle \psi| R(\vec\theta)^\dagger H  R(\vec\theta)| \psi\rangle
$$

To minimize loss, we employ gradient descent method, in which we iteratively set

$$
\vec\theta = \vec\theta - \alpha {\frac{dL}{d\vec{\theta}}}
$$

with $\alpha$ being the learning rate.

The derivative of loss is given as 

$$
\frac{dL}{d\theta_i} = \langle \psi| \frac{dR(\vec\theta)^\dagger}{d\theta_i} H  R(\vec\theta)| \psi\rangle + \langle \psi| R(\vec\theta)^\dagger H  \frac{dR(\vec\theta)}{d\theta_i}| \psi\rangle
$$

The term $\displaystyle \frac{dR}{d\theta_i}$ can be calculated as 

$$
\frac{dR}{d\theta_{xi}} =
\cdots\otimes \left(\frac{d \exp((i/2)~\theta_{xi}\sigma_x)}{d\theta_{xi}}\times \exp((i/2)\theta_{yi}\sigma_y)\times \exp((i/2)\theta_{zi}\sigma_z)\right) \otimes \cdots
$$

which is equal to

$$
= \cdots\otimes \left( \frac{i}{2}\sigma_x\times \exp((i/2)~\theta_{xi}\sigma_x)\times \exp((i/2)\theta_{yi}\sigma_y)\times \exp((i/2)\theta_{zi}\sigma_z)\right) \otimes \cdots
$$

and similarly for $yi$ and $zi$

To calculate $\displaystyle \frac{dR^\dagger}{d\theta_i}$, use the fact that $(A\otimes B)^\dagger = A^\dagger \otimes B^\dagger$, giving

$$
R(\vec\theta)^\dagger = R_1(\vec\theta_1)^\dagger\otimes R_2(\vec\theta_2)^\dagger \otimes \cdots\otimes R_N(\vec\theta_N)^\dagger
$$

and therefore

$$
\frac{dR^\dagger}{d\theta_{xi}} =
\cdots\otimes \left(\frac{d \exp((i/2)~\theta_{xi}\sigma_x)}{d\theta_{xi}}\times \exp((i/2)\theta_{yi}\sigma_y)\times \exp((i/2)\theta_{zi}\sigma_z)\right)^\dagger \otimes \cdots
$$

which is equal to 

$$
= \cdots\otimes \left( \frac{i}{2}\sigma_x\times \exp((i/2)~\theta_{xi}\sigma_x)\times \exp((i/2)\theta_{yi}\sigma_y)\times \exp((i/2)\theta_{zi}\sigma_z)\right)^\dagger \otimes \cdots
$$

Therefore, we have 

$$
\frac{dR^\dagger}{d\theta_i} = \left( \frac{dR}{d\theta_i}  \right)^\dagger
$$

By iteratively obtaining the gradients and setting the $\theta\text{s}$ for the theory, we obtain the ground state of the Schwinger Model.

> TODO #1: Quantify the error in the ground state energy and the angles.

> TODO #2: Quantify the stopping condition for the _training_ loop. Currently, the stopping of the loop is ambiguous. I presumed that when the maximum of all $\theta$ gradients tends to zero, we will have reached the minima (which should be theoretically the case). But the maximum doesnt even reach 1e-3 after 10000 iterations, at which point the energy is barely changing by 1e-4 %, meaning that the loop can be safely terminated without giving rise to significant errors.

> TODO #3: Optimize!

### Particle Density

![pd](gs_pd.png)

### Order Parameter

![op](gs_op.png)

We can see that there is a phase transition around $m_c \approx -0.5$ with the actual critical mass being $m_c \approx -0.7$ (Byrnes, T., Sriganesh, P., Bursill, R. & Hamer, C. Density matrix renormalization  group approach to the massive Schwinger model. Nucl. Phys. B 109, 202–206  (2002). )

