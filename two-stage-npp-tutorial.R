# =============================================================================
#  Two-stage correlated-endpoint NPP: a worked example
#
#  This script walks through ONE analysis of ONE simulated dataset, step by
#  step, so you can see exactly what the method does. It mirrors the
#  simulation code used in the paper, but strips out the trial loop, the
#  scenario grid and the parallel machinery.
#
#  The setting:
#    - A HISTORICAL study measured only an intermediate outcome Y0
#      (e.g. a quality-of-life score like KCCQ).
#    - The CURRENT trial measures both Y0 and the clinical outcome of
#      interest Y1 (e.g. 6-minute walk distance) on every patient.
#    - There is one binary biomarker X (e.g. clonal hematopoiesis yes/no),
#      and we want to know which biomarker subgroup benefits from
#      treatment on Y1.
#
#  The method in two sentences:
#    Stage 1 learns how treatment effects on Y0 translate into treatment
#    effects on Y1, using the paired (Y0, Y1) measurements in the current
#    trial. Stage 2 borrows the historical Y0 data through a normalized
#    power prior (NPP) and pushes that information onto Y1 through the
#    Stage 1 "translation".
#
#  Outline:
#    0. Packages and settings
#    1. Simulate historical and current data
#    2. Stage 1: estimate the surrogate link  phi_j = b_j * theta_j + noise
#    3. Stage 2: fit the joint Bayesian model in Stan
#    4. Read off the results: borrowing weight, subgroup effects, decisions
#    5. (Optional) Compare with an analysis that ignores the historical data
# =============================================================================


# =============================================================================
# 0. PACKAGES AND SETTINGS
# =============================================================================

library(rstan)      # Bayesian model fitting (Stage 2)
library(systemfit)  # Seemingly unrelated regression for (Y0, Y1) (Stage 1)
library(MASS)       # mvrnorm(), for the parametric bootstrap (Stage 1)

rstan_options(auto_write = TRUE)          # cache the compiled Stan model
options(mc.cores = parallel::detectCores())

set.seed(2026)


# =============================================================================
# 1. SIMULATE DATA
# =============================================================================
# Both outcomes follow the same simple linear model with a treatment-by-
# biomarker interaction:
#
#   Y0 = theta_1 + theta_2 * X + theta_3 * Z + theta_4 * X * Z + error
#   Y1 = phi_1   + phi_2   * X + phi_3   * Z + phi_4   * X * Z + error
#
# where Z is treatment (0/1) and X is the biomarker (0/1).
#
# The treatment effect on Y1 in biomarker group x is
#   beta(x) = phi_3 + phi_4 * x
# so beta(0) = phi_3 and beta(1) = phi_3 + phi_4.
# -----------------------------------------------------------------------------

# --- Sample sizes ---
N_hist <- 400   # historical study (Y0 only)
N_curr <- 150   # current trial (Y0 and Y1); think of this as the interim look

# --- True parameters for the intermediate outcome Y0 ---
#     (intercept, biomarker, treatment, biomarker x treatment)
theta_true <- c(0, 0, -0.25, 0.35)
# Treatment effect on Y0: -0.25 if X = 0, +0.10 if X = 1

# --- True surrogate link: phi_j = b_j * theta_j ---
# Here the interaction is twice as strong on Y1 as on Y0 (b_4 = 2).
# This is exactly the kind of mismatch that breaks a naive "Y0 = Y1"
# borrowing approach.
b_true   <- c(1, 1, 1, 2)
phi_true <- b_true * theta_true
# Treatment effect on Y1: beta(0) = -0.25, beta(1) = -0.25 + 0.70 = 0.45
# So only the X = 1 subgroup benefits.

# --- Within-patient correlation between Y0 and Y1 in the current trial ---
rho <- 0.8

# --- Historical study: only Y0 is recorded ---
x_hist  <- rbinom(N_hist, 1, 0.5)
z_hist  <- rbinom(N_hist, 1, 0.5)
y0_hist <- theta_true[1] + theta_true[2] * x_hist +
  theta_true[3] * z_hist + theta_true[4] * x_hist * z_hist +
  rnorm(N_hist, 0, 1)

# --- Current trial: both Y0 and Y1, with correlated errors ---
x_curr <- rbinom(N_curr, 1, 0.5)
z_curr <- rbinom(N_curr, 1, 0.5)

# Draw (error_Y0, error_Y1) pairs from a bivariate normal with correlation rho
err <- mvrnorm(N_curr, mu = c(0, 0),
               Sigma = matrix(c(1, rho, rho, 1), 2, 2))

y0_curr <- theta_true[1] + theta_true[2] * x_curr +
  theta_true[3] * z_curr + theta_true[4] * x_curr * z_curr + err[, 1]
y1_curr <- phi_true[1]   + phi_true[2]   * x_curr +
  phi_true[3]   * z_curr + phi_true[4]   * x_curr * z_curr + err[, 2]

# Design matrix for the historical study: columns match theta_1..theta_4
X_hist <- cbind(1, x_hist, z_hist, x_hist * z_hist)


# =============================================================================
# 2. STAGE 1: ESTIMATE THE SURROGATE LINK
# =============================================================================
# Goal: learn, for each coefficient j, how the Y0 coefficient theta_j maps
# onto the Y1 coefficient phi_j:
#
#     phi_j = b_j * theta_j + eta_j,     eta_j ~ N(0, tau_j^2)
#
# We use ONLY the current trial for this, because it is the only place
# where Y0 and Y1 are measured on the same patients.
#
# Recipe:
#   (a) Fit Y0 and Y1 jointly (seemingly unrelated regression, SUR) to get
#       estimates of theta and phi AND their joint covariance. The joint
#       covariance matters: because Y0 and Y1 are correlated within
#       patients, the errors in theta-hat and phi-hat are correlated too.
#   (b) Parametric bootstrap: draw many (theta, phi) pairs from the
#       estimated sampling distribution.
#   (c) For each j, regress the phi_j draws on the theta_j draws (no
#       intercept). The slope is b_j-hat, the residual variance is
#       tau_j^2-hat, and the slope's variance is Var(b_j-hat).
# -----------------------------------------------------------------------------

# --- (a) Joint fit of Y0 and Y1 in the current trial ---
df_curr <- data.frame(y0 = y0_curr, y1 = y1_curr,
                      x = x_curr, z = z_curr, xz = x_curr * z_curr)

sur_fit <- systemfit(
  list(theta = y0 ~ x + z + xz,     # equation for Y0
       phi   = y1 ~ x + z + xz),    # equation for Y1
  method = "SUR", data = df_curr
)

theta_hat <- coef(sur_fit)[grep("^theta_", names(coef(sur_fit)))]
phi_hat   <- coef(sur_fit)[grep("^phi_",   names(coef(sur_fit)))]
Sigma_hat <- vcov(sur_fit)   # 8 x 8 joint covariance of (theta-hat, phi-hat)

cat("Current-trial estimates (SUR):\n")
print(round(rbind(theta_hat = theta_hat, phi_hat = phi_hat), 3))

# --- (b) Parametric bootstrap ---
R_boot <- 2000
draws <- mvrnorm(R_boot, mu = c(theta_hat, phi_hat), Sigma = Sigma_hat)
boot_theta <- draws[, 1:4]   # bootstrap draws of theta-hat
boot_phi   <- draws[, 5:8]   # bootstrap draws of phi-hat

# --- (c) One no-intercept regression per coefficient ---
b_hat     <- numeric(4)   # estimated slopes b_j
var_b_hat <- numeric(4)   # uncertainty in each slope
tau2_hat  <- numeric(4)   # residual variance tau_j^2

for (j in 1:4) {
  fit_j        <- lm(boot_phi[, j] ~ -1 + boot_theta[, j])
  b_hat[j]     <- coef(fit_j)[1]
  var_b_hat[j] <- vcov(fit_j)[1, 1]
  tau2_hat[j]  <- max(var(residuals(fit_j)), 1e-6)  # guard against zero
}

# --- (Optional) Add external information on the interaction slope b_4 ---
# The interaction slope is usually the hardest to estimate from one trial,
# and the bootstrap regression tends to shrink it toward zero (see the
# Discussion of the paper). If a meta-analysis of past trials reporting both
# outcomes gives an estimate of b_4, we can combine it with the bootstrap
# estimate by a simple normal-normal (precision-weighted) update. This is
# Web Appendix A, equation A.3. Set use_meta_prior <- FALSE to skip it.
use_meta_prior <- TRUE
b4_meta    <- 2.0    # external estimate of b_4
b4_meta_se <- 0.03   # its standard error

if (use_meta_prior) {
  prec_meta  <- 1 / b4_meta_se^2      # precision of external estimate
  prec_boot  <- 1 / var_b_hat[4]      # precision of bootstrap estimate
  b_hat[4]     <- (prec_meta * b4_meta + prec_boot * b_hat[4]) /
    (prec_meta + prec_boot)
  var_b_hat[4] <- 1 / (prec_meta + prec_boot)
}

cat("\nStage 1 surrogate link:\n")
print(data.frame(j = 1:4,
                       coefficient = c("intercept", "biomarker",
                                       "treatment", "biomarker x trt"),
                       b_hat = b_hat, sd_b_hat = sqrt(var_b_hat),
                       tau_hat = sqrt(tau2_hat), true_b = b_true))

# --- Safeguard: no-borrowing rule ---
# If every slope is near zero, Y0 tells us essentially nothing about Y1,
# so we should not borrow at all. In that case, fit Y1 alone (Section 5).
b_threshold <- 0.1
borrow <- any(abs(b_hat) >= b_threshold)
cat(sprintf("\nBorrow from historical data? %s\n", borrow))


# =============================================================================
# 3. STAGE 2: JOINT BAYESIAN MODEL IN STAN
# =============================================================================
# Now we hold the Stage 1 estimates fixed and fit:
#
#   p(theta, phi, delta | data)  proportional to
#       L1(phi | current Y1)                          <- current trial, Y1 only
#     x prod_j N(phi_j | b_j-hat * theta_j, v_j)      <- surrogate prior
#     x L0(theta | historical Y0)^delta / C(delta)    <- normalized power prior
#     x pi0(theta) x pi(delta)                        <- base priors
#
# with  v_j = tau_j^2-hat + theta_j^2 * Var(b_j-hat).
#
# How to read this:
#   - delta in [0, 1] is the borrowing weight. It is LEARNED: if the
#     historical Y0 data are consistent with what the surrogate prior and
#     the current Y1 data imply, delta is pushed up; if they conflict,
#     delta shrinks toward 0 and the historical data are largely ignored.
#   - C(delta) is the normalizing constant that makes the power prior a
#     proper distribution for every delta. Without it, delta cannot be
#     learned sensibly. For a Gaussian Y0 it has a closed form (below).
#   - The current Y0 data are NOT in this model. They were used once, in
#     Stage 1, to learn the link. Using them again here would count the
#     current trial twice.
#   - The historical residual variance is plugged in as a fixed estimate,
#     which keeps the closed-form C(delta) simple.
# -----------------------------------------------------------------------------

stan_code <- "
functions {
  // Log of the NPP normalizing constant C(delta) for a Gaussian linear
  // model with known variance sigma2 and a N(m0, S0) base prior:
  //   C(delta) = integral of  L0(theta)^delta * N(theta | m0, S0)  d theta
  // Completing the square gives this closed form.
  real log_C_delta(real delta, vector y, matrix X,
                   real sigma2, vector m0, matrix S0) {
    int p = rows(m0);
    matrix[p, p] S0_inv = inverse_spd(S0);
    matrix[p, p] A      = S0_inv + (delta / sigma2) * crossprod(X);
    vector[p]    b_vec  = S0_inv * m0 + (delta / sigma2) * (X' * y);
    return
      - 0.5 * delta * rows(y) * log(2 * pi() * sigma2)
      - 0.5 * log_determinant(S0)
      - 0.5 * dot_product(m0, S0_inv * m0)
      - 0.5 * (delta / sigma2) * dot_self(y)
      + 0.5 * dot_product(b_vec, inverse_spd(A) * b_vec)
      - 0.5 * log_determinant(A);
  }
}

data {
  // --- Historical study (Y0 only) ---
  int<lower=1> N_hist;
  vector[N_hist] y0_hist;
  matrix[N_hist, 4] X_hist;        // columns: 1, x, z, x*z
  real<lower=0> sigma2_hist;       // plug-in residual variance of Y0

  // --- Current trial (Y1 only; current Y0 was used in Stage 1) ---
  int<lower=1> N_curr;
  vector[N_curr] y1_curr;
  matrix[N_curr, 4] X_curr;        // columns: 1, x, z, x*z

  // --- Base prior on theta and prior on delta ---
  vector[4] m0;
  matrix[4, 4] S0;
  real<lower=0> a_delta;           // Beta(a, b) prior on delta
  real<lower=0> b_delta;

  // --- Stage 1 surrogate link, held fixed ---
  vector[4] b_hat;
  vector<lower=0>[4] var_b_hat;
  vector<lower=0>[4] tau2_hat;
}

parameters {
  real<lower=0, upper=1> delta;    // borrowing weight
  vector[4] theta;                 // Y0 coefficients (shared across studies)
  vector[4] phi_raw;               // standardized version of phi (see below)
  real<lower=0> sigma_1;           // residual SD of Y1
}

transformed parameters {
  // Surrogate prior:  phi_j | theta_j ~ N(b_hat_j * theta_j, v_j)
  // Written in 'non-centered' form (phi = mean + sd * standard normal),
  // which helps Stan sample efficiently.
  vector[4] phi;
  for (j in 1:4) {
    real v_j = tau2_hat[j] + square(theta[j]) * var_b_hat[j];
    phi[j] = b_hat[j] * theta[j] + sqrt(v_j) * phi_raw[j];
  }
}

model {
  // --- Priors ---
  delta   ~ beta(a_delta, b_delta);
  theta   ~ multi_normal(m0, S0);       // base prior pi0(theta)
  phi_raw ~ std_normal();               // implies the surrogate prior on phi
  sigma_1 ~ normal(0, 10);

  // --- Normalized power prior: historical likelihood raised to delta ---
  target += delta * normal_lpdf(y0_hist | X_hist * theta, sqrt(sigma2_hist));
  target += -log_C_delta(delta, y0_hist, X_hist, sigma2_hist, m0, S0);

  // --- Current-trial likelihood for Y1 ---
  y1_curr ~ normal(X_curr * phi, sigma_1);
}

generated quantities {
  // Treatment effect on Y1 within each biomarker group
  real beta_x0 = phi[3];               // X = 0
  real beta_x1 = phi[3] + phi[4];      // X = 1
}
"

# Plug-in estimate of the historical residual variance
sigma2_hist <- max(var(residuals(lm(y0_hist ~ X_hist - 1))), 1e-6)

stan_data <- list(
  N_hist      = N_hist,
  y0_hist     = y0_hist,
  X_hist      = X_hist,
  sigma2_hist = sigma2_hist,
  N_curr      = N_curr,
  y1_curr     = y1_curr,
  X_curr      = cbind(1, x_curr, z_curr, x_curr * z_curr),
  m0          = rep(0, 4),        # weakly informative base prior:
  S0          = diag(100, 4),     #   theta ~ N(0, 100 I)
  a_delta     = 1,                # delta ~ Beta(1, 1), i.e. uniform
  b_delta     = 1,
  b_hat       = b_hat,
  var_b_hat   = var_b_hat,
  tau2_hat    = tau2_hat
)

# The NPP posterior can have awkward geometry (delta interacts with theta),
# so we use a high adapt_delta to avoid divergent transitions.
fit <- stan(model_code = stan_code, data = stan_data,
            iter = 4000, warmup = 2000, chains = 4, seed = 1,
            control = list(adapt_delta = 0.99, max_treedepth = 15))

# Always check convergence: Rhat close to 1 and adequate effective sample size
print(fit, pars = c("delta", "theta", "phi", "beta_x0", "beta_x1"),
      probs = c(0.025, 0.5, 0.975), digits = 3)


# =============================================================================
# 4. INTERPRET THE RESULTS AND APPLY THE TRIAL DECISION RULES
# =============================================================================

post <- rstan::extract(fit)

# --- How much did we borrow? ---
cat(sprintf("\nPosterior mean borrowing weight E[delta | data] = %.2f\n",
            mean(post$delta)))
# Values near 1: historical data treated almost like current data.
# Values near 0: historical data largely discounted.

# --- Subgroup-specific treatment effects on Y1 ---
summ <- function(draws) c(mean = mean(draws), quantile(draws, c(0.025, 0.975)))
cat("\nTreatment effect on Y1 (posterior mean and 95% CrI):\n")
print(round(rbind(`X = 0` = summ(post$beta_x0),
                  `X = 1` = summ(post$beta_x1)), 3))
cat(sprintf("True values: beta(0) = %.2f, beta(1) = %.2f\n",
            phi_true[3], phi_true[3] + phi_true[4]))

# --- Decision rules from the paper ---
e1    <- 0       # clinically meaningful threshold for beta(x)
alpha <- 0.05    # subgroup inclusion: P(beta(x) > e1) > 1 - alpha
b1    <- 0       # efficacy threshold on the enriched effect Delta
B1    <- 0.985   # stop for efficacy if P(Delta > b1) > B1
B2    <- 0.80    # stop for futility if P(Delta < b1) > B2

# (i) Effective subspace: biomarker groups with strong evidence of benefit
p_benefit <- c(`X = 0` = mean(post$beta_x0 > e1),
               `X = 1` = mean(post$beta_x1 > e1))
cat("\nP(beta(x) > e1 | data):\n"); print(round(p_benefit, 3))

eff_subspace <- c(0, 1)[p_benefit > 1 - alpha]
if (length(eff_subspace) == 0) eff_subspace <- c(0, 1)  # none selected -> keep everyone
cat(sprintf("Effective subspace: X in {%s}\n",
            paste(eff_subspace, collapse = ", ")))

# (ii) Enriched treatment effect Delta: average of beta(x) over the
#      effective subspace, weighted by the observed biomarker mix
p_x1 <- mean(x_curr == 1)
Delta <- if (identical(eff_subspace, 0)) post$beta_x0 else
  if (identical(eff_subspace, 1)) post$beta_x1 else
    (1 - p_x1) * post$beta_x0 + p_x1 * post$beta_x1

# (iii) Stopping decisions at this look
p_eff <- mean(Delta > b1)
p_fut <- mean(Delta < b1)
decision <- if (p_eff > B1) "STOP for efficacy" else
  if (p_fut > B2) "STOP for futility" else
    if (length(eff_subspace) == 1) "CONTINUE, enrolling only the effective subspace" else
      "CONTINUE, enrolling everyone"

cat(sprintf("\nP(Delta > %g) = %.3f   P(Delta < %g) = %.3f\n",
            b1, p_eff, b1, p_fut))
cat(sprintf("Decision at this look: %s\n", decision))
# In a full trial you would repeat Stages 1 and 2 at each look with the
# accumulated data, as described in the Interim Analysis Procedure.


# =============================================================================
# 5. (OPTIONAL) COMPARISON: IGNORE THE HISTORICAL DATA
# =============================================================================
# The no-borrowing analysis fits the Y1 model to the current trial alone.
# With the diffuse priors used in the paper, the posterior is essentially
# centered at the least-squares estimate, so lm() gives a quick picture of
# what borrowing adds. Compare the interval widths with Section 4.

nb_fit <- lm(y1_curr ~ x_curr * z_curr)
nb_est <- c(`X = 0` = coef(nb_fit)["z_curr"],
            `X = 1` = sum(coef(nb_fit)[c("z_curr", "x_curr:z_curr")]))
V  <- vcov(nb_fit)
nb_se <- c(sqrt(V["z_curr", "z_curr"]),
           sqrt(V["z_curr", "z_curr"] + V["x_curr:z_curr", "x_curr:z_curr"] +
                  2 * V["z_curr", "x_curr:z_curr"]))

cat("\nNo borrowing: treatment effect on Y1 (estimate and 95% interval):\n")
print(round(cbind(estimate = nb_est,
                  lower = nb_est - 1.96 * nb_se,
                  upper = nb_est + 1.96 * nb_se), 3))