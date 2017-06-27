#!usr/bin/env python3

import numpy as np
import pylab as pl

def f1(x): return 1/(np.exp(-x**2))

def Fourier_Const(f, xbounds,N):
	L = xbounds[1]-xbounds[0]
	x = np.linspace(xbounds[0],xbounds[1],num = 10000)
	y = f(x)
	a_0 = (1/float(len(x)))*(np.sum(y) - 0.5*(y[0] + y[-1]))
	a_n = np.zeros(N)
	b_n = np.zeros(N)
	for n in range(1,N+1):
		ya = f(x)*np.cos(n*np.pi*x/L)
		a_n[n-1] = (1/float(len(x)))*(np.sum(ya) - 0.5*(ya[0] + ya[-1]))
		yb = f(x)*np.sin(n*np.pi*x/L)
		b_n[n-1] = (1/float(len(x)))*(np.sum(yb) - 0.5*(yb[0] + yb[-1]))
	return a_0,a_n,b_n

def Fourier_Eval(xbounds,a_0,a_n,b_n,N):
	n = np.arange(1,N+1)
	L = xbounds[1]-xbounds[0]
	x = np.linspace(xbounds[0],xbounds[1],num = 1000)
	y = np.zeros_like(x)
	for n in range(1,N+1):
		if n == 1: y += a_0/2
		y+= a_n[n-1]*np.cos(n*np.pi*x/L)
		y+= b_n[n-1]*np.sin(n*np.pi*x/L)
	return x,y

def main():
	L = np.pi
	N = 1
	xbounds = np.array([0,L])
	a_0,a_n,b_n = Fourier_Const(f1,xbounds,N)
	x,y = Fourier_Eval(xbounds,a_0,a_n,b_n,N)
	print(a_0,a_n,b_n)

main()

