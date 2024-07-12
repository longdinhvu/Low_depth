import numpy as np
from tqdm.notebook import tqdm
from poly_approx.approx import kappa_map,tossing_verification,approx_poly,degree_scaled

class low_depth:
    # Implementing theorem 31 of 2207.08628v3
    def __init__(self, epsilon, delta, beta, a,approx_mode, bias_mode,version):
        self.epsilon = epsilon # error
        self.delta = delta # failure probability
        self.beta = beta # low-depth exponent
        self.a = a # the amplitude
        self.approx_mode = approx_mode # 'poly' or 'erf', build the approximating polynomial, or use the erf
        self.bias_mode = bias_mode # 'partial' or 'complete'
        self.version = version # implement some changes to the original algorithm
        
    def _initialize(self):
        # initialize the parameters
        a_min = 0
        a_max = 1
        Delta = a_max-a_min
        a_mid = a_min+Delta/2
        return a_min,a_max,Delta,a_mid
        
    def _set_params(self, a_mid, Delta):
        if self.version == 'old':
            # original algorithm
            if a_mid >= (Delta**(1-self.beta))/2:
                eta = 0.01*(Delta**self.beta)
                tau = 0.01*(Delta**self.beta)
                k = 0.5*kappa_map(tau)/(Delta**(1-self.beta))
                gamma = 0.01*(Delta**self.beta)
                
            else:
                eta = 0.01
                tau = 0.01
                k = 0.5*kappa_map(tau)/Delta
                gamma = 0.01
        else:
            # new algorithm
            if a_mid >= (Delta**(1-self.beta))/2 and 0.01*(Delta**self.beta)<=0.004:
                # adding an extra constraint on Delta**beta
                eta = 0.01*(Delta**self.beta)
                tau = 0.01*(Delta**self.beta)
                k = 2*kappa_map(tau)/(Delta**(1-self.beta)) # modify k by a factor of 4
                gamma = 0.01*(Delta**self.beta)
            else:
                eta = 0.01
                tau = 0.01
                k = 0.5*kappa_map(tau)/Delta
                gamma = 0.01
        return eta,tau,k,gamma
        
    def _coin_toss(self,a_min,a_max,delta, gamma,k,tau,eta):
        #start = time.time()
        P = approx_poly(a_min,a_max,k,tau,eta,approx_mode=self.approx_mode,bias_mode=self.bias_mode)(self.a)
        #end = time.time()
        #print('Time for approx_poly to run: ', end - start)
        #print(P)
        Pr = P*P # toss probability
        m = int(np.ceil(0.5*gamma**(-2)*np.log(1/delta))) # number of tosses
        if self.bias_mode == 'complete':
            pct = Pr
        else:
            #start = time.time()
            pct = np.random.binomial(m,Pr)/m # normalized sum of Bernoulli variables 
            #end = time.time()
            #print('Time for sampling binomial to run: ', end - start)
        n = degree_scaled(k,eta)
        return pct,n,m
        
    def _main(self):
        a_min,a_max,Delta,a_mid = self._initialize()
        T = int(np.ceil(np.emath.logn(0.9,self.epsilon)))
        delta_t = self.delta/T
        d = 0 # max depth
        D = 0 # total queries complexity
        for t in tqdm(range(T)):
            eta,tau,k,gamma = self._set_params(a_mid,Delta)
            if tossing_verification(a_min,a_max,k,tau,eta,gamma,approx_mode=self.approx_mode) == False:
                print('==================')
                print('Failure to satisfy the desired bounds at iteration ',t)
                print('The value of a_min,a_max,k,tau,eta,gamma at this step :', (a_min,a_max,k,tau,eta,gamma))
                print('==================')
            #print(approx_verification(a,a_min,a_max,k,tau,eta,approx_mode=self.approx_mode))
            #start = time.time()
            pct,n,m = self._coin_toss(a_min,a_max,delta_t,gamma,k,tau,eta)
            #end = time.time()
            #print('Time for _coin_toss to run: ', end - start)
            d = max(d,n)
            D += n*m
            if pct>0.25+gamma**2:
                a_min += 0.1*Delta
            else:
                a_max -= 0.1*Delta
            Delta = a_max-a_min
            a_mid = (a_max+a_min)/2
        a_hat = a_mid
        return a_hat, d, D    