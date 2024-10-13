# Numerical simulation of $U(1)$ Wilson Lattice Gauge Model - The Schwinger Model

In this repository, we present the numerical simulations of the Schwinger model, which is a $U(1)$ Wilson lattice gauge theory, by studying the dynamics of the spin-lattice model obtained by the Jordan-Wigner Transformation.

We discuss first, the [real-time evolution](#real-time-dynamics) of particle density, entanglement entropy and electric fields for the vacuum of the Schwinger Model. 

We then also discuss various methods to approximate the ground state of the system, such as [variational method to find the separable product state approximation for ground state](#variational-approximation-of-ground-state---seperable-product-state-approximation), [variational method to find the MPS approximation for the ground state](#variational-approximation-of-ground-state---matrix-product-state)
 and [adiabatic quantum evolution to obtain the ground state](#adiabatic-quantum-evolution) of the model for different bare mass.

We also measure the fidelity of the approximate ground states with the exact ground state, and the phase transitions of the system.

Further, I am working on implementing a [PINN which can be trained to obtain the ground state of the system](#pinn-for-finding-the-ground-state). Preliminarily, I have obtained the phase transition as expected, but there are deviations from the expected values at positive bare masses.

## Real-Time Dynamics

[Jupyter Notebook](num_sim_schwinger.ipynb)

The numerical simulations of the real time dynamics is performed by both exact exponentiation of the Hamiltonian and Trotterization of the Hamiltonian. We present both the results below.

The theory and the experimental results are presented in the paper by Muschik et al.: https://iopscience.iop.org/article/10.1088/1367-2630/aa89ab

The results are as follows, and it can see that it matches the results in the paper.

### Particle Density and Half Chain Entanglement Entropy

![nu_and_s](nu_and_s.png)

### Electric Field

![elec_field](elec_field.png)

## Variational Approximation of Ground State - Seperable Product State Approximation

[Jupyter Notebook](vqe_gs_schwinger.ipynb)

Entanglement entropy, which quantifies the amount of entanglement (correlation) in a multipartite system, is a very good indicator of phase transitions. It is minimal at regions that are far from phase transitions, and increases as one approaches the critical value of the parameter, with it being maximum at the critical parameter. Therefore, one can approximately obtain the ground state as a (separable) product state, for regions far away from the critical point.

In this section, we variationally obtain the product state that best approximates the ground state for a given set of Hamiltonian parameters, and quantitatively observe quantities like particle density, order parameter and entanglement entropy, and also quantify the overlap between the variationally obtained product state and the exact state, and also the ground state energy difference for different parameters.

To obtain the approximate ground state, we parameterize each spin-lattice site by three angles $\theta_x$, $\theta_y$ and $\theta_z$ and obtain the ground state of the Schwinger Model by minimizing the energy of the system. The variational parameters are optimized using the adam method.

> [!Important]
> This method approximates the ground state using only 3N parameters, where N is the number of sites in the lattice, which is a massive improvement over exponential number of parameters required to exactly describe the state. We see that the ground states are approximated to a very good degree by the product states, indicating that the ground state has very less entanglement, except for critical points.
> This method fails for systems whose ground state is a highly entangled state, in which case the Tensor Network methods produce better results.
> The presented method is equivalent to representing the ground state by a Matrix Product State (MPS), with bond order D=1. Therefore, as expected, the entanglement entropy of the state obtained in the variational method is zero.

The experimental results are presented in the paper by Kokail et al.: https://www.nature.com/articles/s41586-019-1177-4

### Gradient Calculation for optimization algorithm

> Please write to me if you find any error in the following analysis

We start with a random state as the initial state and we choose to rotate the state at each site by three angles $\theta_x$ $(R_x)$, $\theta_y$ $(R_y)$, & $\theta_z$ $(R_z)$

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

### Results

I have implemented, so far, the gradient descent, stochastic gradient descent and adam optimizers, with cosine annealing and exponential learning rate schedulers. The results below are the results for adam optimizer with cosine annealed learning rate, and with injected noise.

The noise injection and cosine annealing of the learning rate is done so that the optimizer can visit a huge portion of the configuration space, and not get stuck in a local minima.

> TODO #1: Quantify the error in the ground state energy and the angles.

> TODO #2: Stopping condition for the gradient descent has been set to `max(gradient) < 1e-5`. Verify if this stopping condition is sufficient.

The following are the results obtained. (`Exact` stands for values obtained from the exact diagonalization of the Hamiltonian)

### Particle Density

![pd](gs_pd.png)

### Order Parameter

![op](gs_op.png)

We can see that there is a phase transition around $m_c \approx -0.7$ `(Byrnes, T., Sriganesh, P., Bursill, R. & Hamer, C. Density matrix renormalization  group approach to the massive Schwinger model. Nucl. Phys. B 109, 202–206  (2002). )`

We also see that for negative bare mass, it is energetically favourable to have particle excitations, which leads to non vanishing ground state particle density.

### Ground State Energy

![gs_energy](gs_energy.png)

We see that the ground state energy is symmetric around $m_c \approx -0.7$, and the difference between the exact energy and the approximate energy grows with $N$.

### Entanglement Entropy

![ee](gs_entanglement.png)

As expected from the phase transitions, entanglement entropy at masses away from $m_c$ is close to zero, therefore indicating that the obtained product states are very good approximations for the exact ground state, while for $m$ close to $m_c$, the entanglement entropy is large, indicating that the product states are not good approximations for the exact ground state.

### Wavefunction Overlap

![overlap](gs_overlap.png)

We see that for $m$ away from $m_c$, the overlap is close to 1, and for $m$ close to $m_c$, the overlap is close to 0, which is expected.

### Energy Difference

![ed](gs_energy_diff.png)

## Variational Approximation of Ground State - Matrix Product State

[Jupyter Notebook](tn_schwinger.ipynb)

A more faithful, yet efficient representation of the quantum state for the many body system is a Matrix Product State (with closed boundary), which is a set of $Nd$ $D\times D$ matrices $A_n^i$, such that there is one matrix for every local Hilbert space dimension at every lattice site.

The wavefunction of the system is defined then, in terms of the matrix products, as

$$
|\psi\rangle = \sum \mathrm{tr}(A_1^{i_1}~A_2^{i_2}~\cdots ~A_N^{i_n})~|i_1\rangle\otimes | i_2\rangle\otimes \cdots \otimes |i_N\rangle
$$

The matrices have $NdD^2 (\times 2)$ complex (real) parameters, and here I implement gradient descent for these parameters, to minimize the expectation value of the Hamiltonian.

To optimize the inner product $\displaystyle \frac{\langle \psi | H | \psi \rangle}{\langle \psi|\psi\rangle}$, we perform optimization using gradient descent, using

$$
d \left ( \frac{\langle \psi | H | \psi \rangle}{\langle \psi|\psi\rangle} \right) = \frac{\langle \psi|\psi\rangle~(~ d\langle\psi|~ H |\psi \rangle + \langle\psi |H ~d|\psi \rangle ~) - \langle \psi | H | \psi  \rangle ~(~ d\langle\psi|~ |\psi \rangle + \langle\psi | ~d|\psi \rangle ~)}{\langle \psi|\psi\rangle^2}
$$

where $d$ is short for derivative with respect to a given parameter.

Each matrix in the MPS, $A_n^i$ has parameters $`(A_{n}^{i})_{11}^r + i (A_{n}^{i})_{11}^i,  ~ (A_{n}^{i})_{12}^r + i(A_{n}^{i})_{12}^i, \cdots,  ~(A_{n}^{i})_{DD} + i(A_{n}^{i})_{DD}^i`$

Therefore for each parameter,

$`
\displaystyle \frac{d}{d(A_n^i)_{kl}}|\psi(\cdots, (A_n^i)_{kl}, \cdots)\rangle = \sum \left[ \mathrm{tr}\left( \cdots \frac{d A_n^i}{d (A_n^i)_{kl}} \cdots  \right)  |i_1\rangle\otimes | i_2\rangle\otimes \cdots \otimes |i_N\rangle ~~~~\text{if the basis at site} ~n~ \text{is} ~i~\text{,}~0~\text{otherwise}\right]
`$

The derivative of the matrix $A_n^i$ with respect to the parameter $(A_n^i)_{kl}$, is given by

$$
A_{pq} = \begin{cases}
	1(i)~~~\text{    if }k=p, l=q~~~~(i\text{ when we are differentiating with respect to the imaginary part})\\
	0~~~~~~~\text{elsewhere}
\end{cases}
$$

By repeatedly obtaining the gradients and setting the parameters of the MPS to descend along the gradient, we can get a very good approximation of the ground state of the system.

As an example, we have considered here two cases, $N=6$ and $N=8$, both with bond dimension $D=2$. In the first case, the actual number of (complex) parameters needed to specify the system is $2^6 = 64$, while the MPS has $6\times 2\times 2^2 = 48$ parameters. In the second case, the actual number of parameters needed is $256$, while the MPS needs only $64$ parameters. 

We also see that even with such a small number of parameters, we still obtain very close approximations to the ground state, as can be seen from the results below. 

### Particle Density

![ad_pd](gs_pd_tn.png)

### Order Parameter

![ad_op](gs_op_tn.png)

### Ground State Energy

![ad_energy](gs_energy_tn.png)

### Entanglement Entropy

![ad_ee](gs_entanglement_tn.png)

### Wavefunction Overlap

![ad_overlap](gs_overlap_tn.png)

### Energy Difference

![ad_ed](gs_energy_diff_tn.png)

Therefore, we see that for even small bond dimension $D=2$, which is only a marginal improvement from the $0$ entanglement case, we get a very good approximation for the ground state. This is due to the fact that even with $D=2$, we can get entanglement entropies upto $2\log_2(2) = 2$, which is still less than the exact ground state entanglement entropy. Therefore, one can safely assume that since the ground state has very low entanglement, even around critical points, the MPS with very low $D$ can also efficiently represent it.


## Adiabatic Quantum Evolution

[Jupyter Notebook](adiabatic_evolution_gs_schwinger.ipynb)

The adiabatic quantum evolution method is a method to obtain the exact ground state of a system. In the adiabatic quantum evolution method to obtain the ground state of the Schwinger Model, we consider a time dependent Hamiltonian

$$
H(t) = \alpha(t) ~H_{\text{Schwinger}} + \beta(t) ~H_{\text{driving}}
$$

where $\alpha + \beta = 1$ and the driving Hamiltonian is a simple hamiltonian that does not commute with the Schwinger Hamiltonian.

The $\alpha$ and $\beta$ are chosen to be function of time such that $\alpha(0) \ll \beta(0)$ and $\alpha(T) \gg \beta(T)$, so that the system initially starts out dominated by the driving Hamiltonian and ends up dominated by the Schwinger Hamiltonian.

The initial state of the system is chosen to be the ground state of the driving Hamiltonian. We then let the state to evolve according to the time dependent Hamiltonian. If the Hamiltonian varies slowly, such that the time evolution of the state takes place adiabatically, then the state will remain in the ground state of the time dependent Hamiltonian, and therefore, the final state will be the ground state of the Schwinger Hamiltonian.

In our simulation, we choose the $\alpha$ and $\beta$ to change by `1/num_steps` for every one second of evolution of the state. We then obtain the ground state of the Schwinger Model by evolving the state for a total time of `num_steps` seconds.

We see that it outperforms the variational method in terms of computational requirements. It also manages to reach the entangled ground states, which were not accessible by the variational method.

The results are as follows:

### Particle Density

![ad_pd](gs_pd_adiabatic.png)

### Order Parameter

![ad_op](gs_op_adiabatic.png)

### Ground State Energy

![ad_energy](gs_energy_adiabatic.png)

### Entanglement Entropy

![ad_ee](gs_entanglement_adiabatic.png)

### Wavefunction Overlap

![ad_overlap](gs_overlap_adiabatic.png)

### Energy Difference

![ad_ed](gs_energy_diff_adiabatic.png)

We see that the parameters of the adiabatic evolution needs to be further refined for the points very close to the critical point. At other places, the given adiabatic evolution method is able to obtain the exact ground state of the Schwinger Model.

## PINN for finding the ground state

[Jupyter Notebook](PINN_schwinger-in_progress.ipynb)

The PINN takes as input the number $n \in \{1, 2, \cdots, 2^N\}$, indicating the basis $| n \rangle^\dagger = (0,0,\cdots,1_\text{at n}, \cdots, 0,0)$ and outputs the complex amplitude corresponding to the basis.

The training is done by taking the expectation value of the hamiltonian as the loss function, to obtain the state where energy is minimum.

> TODO #1: Fix the problem with convergence. Current model does not exactly converge, and still displays fluctuations even after large epochs.

> TODO #2: Quantify the stopping condition. Current model runs the training loop without stopping at any condition.

The approximate ground state particle densities obtained by this preliminary model is as follows:

![gs_pd_pinn](gs_pd_pinn.png)
