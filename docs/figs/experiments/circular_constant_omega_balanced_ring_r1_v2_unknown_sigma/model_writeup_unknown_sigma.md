# circular_constant_omega_balanced_ring_r1_v2_unknown_sigma — concise model write-up

## 1) Latent dynamics
State is z_t=(r_t, theta_t) with Cartesian embedding x_t=r_t cos(theta_t), y_t=r_t sin(theta_t).

r_(t+1) = clip(r_t + u_t + eps_t, r_min, r_max)
theta_(t+1) = (theta_t + omega*dt) mod 2pi

Control uses an information gradient:
u_t = clip(k * d/dr I_comb(r_t, sigma_hat_t) - beta * u_(t-1), [-u_max, u_max]).

## 2) Observation model and unknown parameter
Observation:
y_t = g * exp(-r_t^2/(2*sigma_c^2)) + eta_t,  eta_t ~ N(0, sigma_y^2).
Unknown parameter is sigma_c (sensor radial width).

## 3) Information terms
I_dyn(r) = C_d - a_d*(r-r_d)^2
I_obs(r; sigma_c) = (r^2/sigma_c^4)*exp(-r^2/sigma_c^2)
I_comb(r; sigma_c) = w_dyn*I_dyn(r) + w_obs*I_obs(r; sigma_c)

## 4) Parameter inference/update
A posterior on sigma_c is maintained on a fixed grid:
p_t(sigma) proportional to p_(t-1)(sigma) * exp(-(y_t - mu(r_t,sigma))^2/(2*sigma_y^2)),
with mu(r,sigma)=g*exp(-r^2/(2*sigma^2)).
The online estimate is posterior mean sigma_hat_t = E[sigma | y_1:t].

## 5) Why/when the combined map changes
I_obs explicitly depends on sigma_c, so changes in sigma_hat_t reshape I_obs and therefore I_comb.
Hence the combined map is time-varying during rollout as inference progresses.

## 6) Quantitative checks (this run)
- true sigma_c: 1.3500
- final sigma_hat: 1.3449
- absolute estimation error: 0.0051
- map-change L2 (t0->tmid): 0.017770
- map-change L2 (tmid->tend): 0.001800
- map-change L2 (t0->tend): 0.015982
- argmax-radius shift (t0->tend): 0.000000
