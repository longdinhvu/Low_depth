from scipy.special import ive
import numpy as np
import math
import mpmath
from scipy.special import eval_chebyt
from scipy.special import erfinv, erf
import matplotlib.pyplot as plt


def kappa_map(tau):
    # Building kappa function as per Fact 27 of 2207.08628v3
    # also equivalent to lemma 10 in 1707.05391
    return np.sqrt(0.5*np.log(2/(np.pi*tau**2)))
    
def tau_map(kappa):
    return  np.sqrt(2/np.pi)*np.exp(-(kappa**2))
    
def degree(k,epsilon):
    # the degree of the approximating polynomial, taken as in Lemma 13 and 14 of 1707.05391
    b = (k**2)/2 #corresponding to the change of variable beta = k^2/2
    tmp = np.ceil(max(b*np.exp(2),np.log(2/epsilon)))
    deg = np.sqrt(2*tmp*np.log(4/epsilon))
    deg = int(np.ceil(deg))
    return deg+deg%2-1
    
def approx_erf(x,k,epsilon,approx_mode='poly'):
    # Implementing equation (70) of https://arxiv.org/pdf/1707.05391
    # building a polynomial approximating the error function erf(kx)
    # epsilon is the absolute error
    # approx_mode = 'poly' or 'erf'
    # if approximation_mode is 'poly', build the approximating polynomial as in the paper
    # else just use the error function for testing other features of the low-depth algorithm
    if approx_mode == 'erf':
        return erf(k*x)
    pre_factor = 2*k/np.sqrt(np.pi)
    b = (k**2)/2
    def summand(j,b,x):
        # Chebyshev polynomials expansion, the exponential factor has been absorbed into ive special function
        output = ive(j,b)*((-1)**j)*(eval_chebyt(2*j+1,x)/(2*j+1)-eval_chebyt(2*j-1,x)/(2*j-1))
        return output
    first_term = ive(0,b)*x
    n = degree(k,epsilon)
    s = 0
    for j in range(1,1+(n-1)//2):
        s += summand(j,b,x)
    output = first_term+s
    output *= pre_factor
    return output
    
def approx_erf_scaled(x,k,eta,approx_mode):
    # Scaling x by 1/2 and k by 2 to expand the domain to [-2,2] as in Lemma 25 of 2207.08628v3
    return approx_erf(x/2,k*2,eta,approx_mode)
    
def degree_scaled(k,eta):
    return degree(k*2,eta)

def approx_poly(a_min,a_max,k,tau,eta,approx_mode='poly',bias_mode='partial'):
    # Implementing the construction in Lemma 29 of 2207.08628v3
    # added a bias_mode parameter to simulate an idealized coin toss
    # if bias_mode == 'complete', output a completely biased coin i.e. 0 on the left half and 1 on the right half
    # of the interval [a_min,a_max]. Else, output the construction
    a_mid = (a_min+a_max)/2
    if bias_mode == 'complete':
        P = lambda x: (1+np.sign(x-a_mid))//2
    else:
        f_0 = lambda x: (1+approx_erf_scaled(x,k,eta,approx_mode)+eta)/(4*eta+tau+2)
        f = lambda x: f_0(x-a_mid)
        P = lambda x: f(x)+f(-x)
    return P
    
def approx_verification(a,a_min,a_max,k,tau,eta,approx_mode='poly',display=False):
    # Check if Lemma 29 is satisfied for a certain set of parameters
    a_mid = (a_min+a_max)/2
    P = approx_poly(a_min,a_max,k,tau,eta,approx_mode,bias_mode='partial')
    upper_bound = lambda x: 0.5+0.11*k*(x-a_mid)+eta+0.25*tau # upper bound on the left segment
    lower_bound = lambda x: 0.5+0.11*k*(x-a_mid)-eta-0.25*tau # lower bound on the right segment
    if display:
        # plot the polynomial and the bounds for visualisation
        x = np.linspace(a_min,a_max,2**10)
        y = P(x)
        fig,ax = plt.subplots(figsize=(10,6))
        ax.plot(x,y)
        first_half = np.linspace(a_min,a_mid,2**9)
        upper = [upper_bound(i) for i in first_half]
        ax.plot(first_half,upper)
        second_half = np.linspace(a_mid,a_max,2**9)
        lower = [lower_bound(i) for i in second_half]
        ax.plot(second_half,lower)
        ax.legend(['Approximating Polynomial','Left upper bound','Right lower bound'])
    if a <= a_mid:
        return P(a) <= upper_bound(a)
    else:
        return P(a) >= lower_bound(a)

def tossing_verification(a_min,a_max,k,tau,eta,gamma,n_points=2**4,approx_mode='poly',display=False):
    # Check if the bounds (151)-(160) are satisfied
    Delta = a_max-a_min
    left = np.linspace(a_min,a_min+0.1*Delta,n_points)
    right = np.linspace(a_max-0.1*Delta,a_max,n_points)
    P = approx_poly(a_min,a_max,k,tau,eta,approx_mode,bias_mode='partial')
    max_left = max([P(a) for a in left])
    min_right = min([P(a) for a in right])
    lower_bound = 0.5+gamma
    upper_bound = 0.5-gamma
    assertion = (max_left<=upper_bound) and (min_right>=lower_bound)
    if display:
        x = np.linspace(a_min,a_max,2**4)
        y = P(x)
        fig,ax = plt.subplots(figsize=(10,6))
        ax.plot(x,y)
        ax.plot(left,[upper_bound]*len(left))
        ax.plot(right,[lower_bound]*len(right))
        ax.legend(['Approximating Polynomial','Left upper bound','Right lower bound'])
    return assertion

def poly_plot(a_min,a_max, k, tau, eta,approx_mode='poly'):
    # test the construction of the approximating polynomial
    # plotting f_0,f and P to better understanding their properties
    a_mid = (a_min+a_max)/2
    
    f_0 = lambda x: (1+approx_erf_scaled(x,k,eta,approx_mode)+eta)/(4*eta+tau+2)
    f = lambda x: f_0(x-a_mid)
    P = lambda x: f(x)+f(-x)

    x = np.linspace(-1,1,2**10)
    f_0val = f_0(x)
    fval = f(x)
    Pval = P(x)
    fig,ax = plt.subplots(figsize=(10,6))
    ax.plot(x,f_0val)
    ax.plot(x,fval)
    ax.plot(x,Pval)
    ax.legend(['f_0','f','P'])
    return f_0,f,P