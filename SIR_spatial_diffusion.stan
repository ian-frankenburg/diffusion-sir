functions {
  real[] diffusion_sir(real t, real[] y, real[] theta,
           real[] x_r, int[] x_i) {
    int n = x_i[1];
    matrix[n,n] S;
    matrix[n,n] I;
    matrix[n,n] dS;
    matrix[n,n] dI;
    int N = x_i[2];
    real beta = theta[1];
    real gamma = theta[2];
    real alpha1 = theta[3];
    real alpha2 = theta[4];
    real d = x_r[1];
    int k=1;
    S = to_matrix(rep_array(0.0,n*n),n,n);
    I = to_matrix(rep_array(0.0,n*n),n,n);
    for(i in 1:n){
      for(j in 1:n){
        S[j,i] = y[k];
        I[j,i] = y[k+n*n];
        k = k + 1;
      }
    }
    dS = to_matrix(rep_array(0.0,n*n),n,n);
    dI = to_matrix(rep_array(0.0,n*n),n,n);
    for(i in 2:(n-1)){
      for(j in 2:(n-1)){
        dS[i,j] = -beta*S[i,j]*I[i,j]/N+alpha1/d^2*(S[i+1,j]+S[i-1,j]+S[i,j+1]+S[i,j-1]-4*S[i,j]);
        dI[i,j] = beta*S[i,j]*I[i,j]/N-gamma*I[i,j]+alpha2/d^2*(I[i+1,j]+I[i-1,j]+I[i,j+1]+I[i,j-1]-4*I[i,j]);
      }
    }
    return append_array(to_array_1d(dS),to_array_1d(dI));
  }
}
data {
  int timelength;
  int popsize;
  int t0;
  real d;
  // gridsize plus boundary
  int gridsize;
  real cases[timelength*(gridsize-2)*(gridsize-2)]; // observed case counts over lattice over time
  matrix[gridsize,gridsize] S0;
  matrix[gridsize,gridsize] I0;
}
transformed data {
  real x_r[1] = {d};
  int x_i[2] = {gridsize,popsize};
  real ts[timelength];
  real y0[2*gridsize*gridsize];
  // for use if using BDF solve instead of RK45
  real rel_tol=10e-5;
  real abs_tol=10e-5;
  int max_num_steps=100000;
  for(i in 1:timelength){
    ts[i] = i;
  }
  y0 = append_array(to_array_1d(S0),to_array_1d(I0));
}
parameters {
  real<lower=0,upper=1> beta;
  real<lower=0,upper=1> gamma;
  real<lower=0> sigma;
  real<lower=0,upper=.01> alpha;

}
transformed parameters{
  real ode[timelength,2*gridsize*gridsize] = rep_array(0.0,timelength,2*gridsize*gridsize);
  //ode = integrate_ode_bdf(diffusion_sir, y0, t0, ts, {beta, gamma, 0, alpha}, x_r, x_i, rel_tol,  abs_tol,  max_num_steps);
  ode = integrate_ode_rk45(diffusion_sir, y0, t0, ts, {beta, gamma, 0, alpha}, x_r, x_i);

}
model{
  matrix[gridsize,gridsize] mat[timelength];
  matrix[gridsize-2,gridsize-2] smallMat[timelength];
  real temparr[gridsize*gridsize];
  int c;
  int r;
  int k;
  // need to loop over the ODE output to build matrix
  for(t in 1:(timelength)){
    temparr = ode[t,(gridsize*gridsize+1):(2*gridsize*gridsize)];
    for(i in 0:(gridsize*gridsize-1)){
      c = i/gridsize;
      r = i%gridsize;
      mat[t,r+1,c+1] = temparr[i+1];
    }
  }
  for(t in 1:timelength){
    for(i in 1:(gridsize-2)){
      for(j in 1:(gridsize-2)){
        // indices will need to be correct to represent lattice structure over time
        smallMat[t,i,j] = mat[t,i+1,j+1];
      }
    }
  }
  sigma ~ inv_gamma(1,1);
  alpha ~ beta(1,50);
  gamma ~ beta(1,4);
  beta ~ beta(4,1);

  // now i have an array of each lattice that holds the numerical solution to the PDEs
  // i just need to make sure the big list of sampled points corresponds to these internal points in smallMat
  k=1;
  for(t in 1:timelength){
    for(i in 1:(gridsize-2)){
      for(j in 1:(gridsize-2)){
        // indices will need to be correct to represent lattice structure over time
        cases[k] ~ normal(smallMat[t,j,i],sigma);
        k=k+1;
      }
    }
  }
}
generated quantities {
  matrix[gridsize,gridsize] mat[timelength];
  matrix[gridsize-2,gridsize-2] smallMat[timelength];
  real temparr[gridsize*gridsize];
  int c;
  int r;
  int k=1;
  real pred_cases[timelength*(gridsize-2)*(gridsize-2)]; // observed case counts over lattice over time
  real ode_pred[timelength,2*gridsize*gridsize] = rep_array(0.0,timelength,2*gridsize*gridsize);
  //ode_pred = integrate_ode_bdf(diffusion_sir, y0, t0, ts, {beta, gamma, 0, alpha}, x_r, x_i, rel_tol,  abs_tol,  max_num_steps);
  ode_pred = integrate_ode_rk45(diffusion_sir, y0, t0, ts, {beta, gamma, 0, alpha}, x_r, x_i);
  for(t in 1:(timelength)){
    temparr = ode_pred[t,(gridsize*gridsize+1):(2*gridsize*gridsize)];
    for(i in 0:(gridsize*gridsize-1)){
      c = i/gridsize;
      r = i%gridsize;
      mat[t,r+1,c+1] = temparr[i+1];
    }
  }
  for(t in 1:timelength){
    for(i in 1:(gridsize-2)){
      for(j in 1:(gridsize-2)){
        // indices will need to be correct to represent lattice structure over time
        smallMat[t,i,j] = mat[t,i+1,j+1];
      }
    }
  }
  // now i have an array of each lattice that holds the numerical solution to the PDEs
  // just need to make sure the big list of sampled points corresponds to these internal points in smallMat
  for(t in 1:timelength){
    for(i in 1:(gridsize-2)){
      for(j in 1:(gridsize-2)){
        pred_cases[k] = normal_rng(smallMat[t,j,i],sigma);
        k=k+1;
      }
    }
  }
}
