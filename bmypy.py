#!usr/bin/env python3

"""
mypy

Matthew Del Favero's collection of useful python functions

Version 1.1

"""

import math
import numpy as np
import pylab as pl
from scipy.stats import chi2 as scichi
import scipy.odr as odr

######## Misc ########

def Simple_Import_Numpy(infile):
	fin = open(infile,'r')
	lines = 0
	for line in fin:
		if lines ==0:
			columns = len(line.split())
		lines += 1
	fin.close()

	fin = open(infile,'r')
	A = np.zeros((lines,columns))
	count = 0
	for line in fin:
		A[count] = np.array(line.split()).astype(float)
		count += 1
	return A

def Simple_Sideways_Import(infile):
	A = Simple_Import_Numpy(infile)
	return np.transpose(A)

def Quicksave(outfile,A):
	A = np.asarray(A)
	fout = open(outfile,'w')
	if len(A.shape) == 0:
		fout.write(str(A) + '\n')
	if len(A.shape) == 1:
		for i in range(len(A)):
			fout.write(str(A[i]) + '\n')
	elif len(A.shape) == 2:
		for i in range(len(A)):
			for j in range(len(A[0])):
				fout.write(str(A[i,j])+' ')
			fout.write('\n')
	fout.close()

def Quicksave_Column(outfile,A):
	A = np.asarray(A)
	fout = open(outfile,'w')
	if len(A.shape) == 0:
		fout.write(str(A) + '\n')
	if len(A.shape) == 1:
		for i in range(len(A)):
			fout.write(str(A[i]) + '\n')
	elif len(A.shape) == 2:
		for i in range(len(A[0])):
			for j in range(len(A)):
				fout.write(str(A[j,i])+' ')
			fout.write('\n')
	fout.close()
		
def PDE_FD_1p(x,y):
	x = np.asarray(x)
	y = np.asarray(y)
	if len(x) != len(y): raise Exception("Finite Difference: X and Y are not the same length!")
	dy = np.zeros_like(y)
	for i in range(len(x)-1):
		dy[i] = (y[i+1] - y[i])/(x[i+1] - x[i])
	return(dy)

######## Plotting ########

def Wire_Mesh_3D(x,y,z):
	fig = pl.figure()
	ax = fig.add_subplot(1,1,1,projections = '3d')
	X,Y = np.meshgrid(X,Y)
	ax.plot_wireframe(X,Y,Z)
	pl.show()
	
def Error_Fit_Plot(x,dx,y,dy,m,b, zero = "no" ,title = "No title", xlabel = "No X label", ylabel = "No Y label",savename = "no"):

	if len(np.array([dx])) == 1: dx = dx*np.ones(len(x))
	if len(np.array([dy])) == 1: dy = dy*np.ones(len(y))
	xlims = np.array([np.min(x),np.max(x)])
	ylims = np.array([np.min(y),np.max(y)])
	Lx = np.max(x) - np.min(x)
	Ly = np.max(y) - np.min(y)
	xlims[0] -= 0.1*Lx + np.max(dx)
	xlims[1] += 0.1*Lx + np.max(dx)
	ylims[0] -= 0.1*Ly + np.max(dy)
	ylims[1] += 0.1*Ly + np.max(dy)
	if zero == "yes":
		xlims[0] = -np.max(dx)
		ylims[0] = -np.max(dy)
	Y0 = m*xlims[0]+b
	Y1 = m*xlims[1]+b
	fig, ax = pl.subplots()
	ax.errorbar(x,y,xerr=dx,yerr=dy,fmt='ro')
	ax.plot(xlims,[Y0,Y1])
	ax.set_title(title)
	ax.set_xlim(xlims)
	ax.set_ylim(ylims)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if savename != "no": pl.savefig(savename)

def Quick_Grid_Plot_To_File(grid,outfile = "redwop.png",color = "gray"):
	pl.figure()
	pl.imshow(grid,cmap=color)
	pl.savefig(outfile)
	pl.close()

def Quick_Hist(x,delimeter,outfile = "redwop.png",xword = "No X Label", yword = "No Y Label",title = "No Title"):
	nbins = (np.max(x)-np.min(x))/delimeter
	pl.hist(x,nbins,facecolor='green',alpha=0.66)
	pl.xlabel(xword)
	pl.ylabel(yword)
	pl.title(title)
	pl.grid(True)
	pl.savefig(outfile)
	pl.close()

######## Error Management ########

def rms(A): return np.sqrt(np.mean(np.square(A)))

def std(A): return np.sqrt(np.sum((A-np.mean(A))**2))

def Err_Anal(x,dx,y,dy):

	if len(x) != len(y): raise Exception("The vectors are not the same length! ")
	if len(x) < 3: raise Exception("Use at least 3 points! ")
	dx = np.array([dx])
	dy = np.array([dy])
	
	N = len(x)
		
	if len(dx) == 1: dx = dx*np.ones(N)
	if len(dy) == 1: dy = dy*np.ones(N)
	sigma = np.sqrt(dx**2 + dy**2)
	
	sigx = np.sqrt(np.sum(dx**2)/N)
	sigy = np.sqrt(np.sum(dx**2)/N)
	sx = sigx*np.sqrt(N/(N-1))
	sy = sigy*np.sqrt(N/(N-1))
	Sx = sx/np.sqrt(N)
	Sy = sy/np.sqrt(N)
	SSx = Sx/np.sqrt(N-2)
	SSy = Sy/np.sqrt(N-2)

	ux = (np.sum(x*dx**(-2)))/(np.sum(dx**-2))
	uy = (np.sum(y*dy**(-2)))/(np.sum(dy**-2))
	sigux = np.sqrt(np.sum(dx**(-2)))
	siguy = np.sqrt(np.sum(dy**(-2)))

	return sigx, sx, Sx, SSx, sigy, sy, Sy, SSy

def ODR_Linear_Fit_f(B,x): return B[0]*x + B[1]

def ODR_Linear_Fit_Yerr(X,Y,dY):

	if len(X) != len(Y): raise Exception("The vectors are not the same length! ")
	if len(X) < 3: raise Exception("Use at least 3 points! ")
	dY = np.array([dY])
	
	N = len(X)
		
	if len(dY) == 1: dY = dY*np.ones(N)

	linear = odr.Model(ODR_Linear_Fit_f)
	mydata = odr.RealData(X,y=Y,sy=dY)
	myodr = odr.ODR(mydata,linear,beta0 = [1,2])
	myoutput = myodr.run()
	A = myoutput.pprint()

	m = myoutput.beta[0]
	b = myoutput.beta[1]
	sm = myoutput.sd_beta[0] 
	sb = myoutput.sd_beta[1] 
	rchi2 = myoutput.res_var
	dof = len(myoutput.beta)
	chi2 = rchi2*dof
	P = 1 - scichi.cdf(chi2,dof)
	return m, sm, b, sb, rchi2, P

def ODR_Linear_Fit(X,dX,Y,dY):

	if len(X) != len(Y): raise Exception("The vectors are not the same length! ")
	if len(X) < 3: raise Exception("Use at least 3 points! ")
	dX = np.array([dX])
	dY = np.array([dY])
	
	N = len(X)
		
	if len(dX) == 1: dX = dX*np.ones(N)
	if len(dY) == 1: dY = dY*np.ones(N)

	linear = odr.Model(ODR_Linear_Fit_f)
	mydata = odr.RealData(X,y=Y,sx=dX,sy=dY)
	myodr = odr.ODR(mydata,linear,beta0 = [1,2])
	myoutput = myodr.run()
	myoutput.pprint()
	
	m = myoutput.beta[0]
	b = myoutput.beta[1]
	sm = myoutput.sd_beta[0] 
	sb = myoutput.sd_beta[1] 
	rchi2 = myoutput.res_var
	dof = len(myoutput.beta)
	chi2 = rchi2*dof
	P = 1 - scichi.cdf(chi2,dof)
	return m, sm, b, sb, rchi2, P

def FD_Err_x(f,x,dx):
	y = f(x)
	sigy = 0.5*(abs(f(x+dx) - f(x)) + abs(f(x-dx) - f(x)))
	return y, sigy

def FD_Err_xc(f,x,dx,c):
	y = f(x,c)
	sigy = 0.5*(abs(f(x+dx,c) - f(x,c)) + abs(f(x-dx,c) - f(x,c)))
	return y, sigy

def FD_Err_xy(f,x,dx,y,dy):
	z = f(x,y)
	sigx = 0.5*(abs(f(x+dx,y) - f(x,y)) + abs(f(x-dx,y) - f(x,y)))
	sigy = 0.5*(abs(f(x,y+dy) - f(x,y)) + abs(f(x,y-dy) - f(x,y)))
	sigz = np.sqrt(sigx**2 + sigy**2)
	return z, sigz

def FD_Err_xyc(f,x,dx,y,dy,c):
	z = f(x,y,c)
	sigx = 0.5*(abs(f(x+dx,y,c) - f(x,y,c)) + abs(f(x-dx,y,c) - f(x,y,c)))
	sigy = 0.5*(abs(f(x,y+dy,c) - f(x,y,c)) + abs(f(x,y-dy,c) - f(x,y,c)))
	sigz = np.sqrt(sigx**2 + sigy**2)
	return z, sigz

def FD_Err_xyz(f,x,dx,y,dy,z,dz):
	val = f(x,y,z)
	sigx = 0.5*(abs(f(x+dx,y,z) - f(x,y,z)) + abs(f(x-dx,y,z) - f(x,y,z)))
	sigy = 0.5*(abs(f(x,y+dy,z) - f(x,y,z)) + abs(f(x,y-dy,z) - f(x,y,z)))
	sigz = 0.5*(abs(f(x,y,z+dz) - f(x,y,z)) + abs(f(x,y,z-dz) - f(x,y,z)))
	sigval = np.sqrt(sigx**2 + sigy**2 + sigz**2)
	return val, sigval

def FD_Err_wxyz(f,w,dw,x,dx,y,dy,z,dz):
	val = f(w,x,y,z)
	sigw = 0.5*(abs(f(w+dw,x,y,z) - f(w,x,y,z)) + abs(f(w-dw,x,y,z) - f(w,x,y,z)))
	sigx = 0.5*(abs(f(w,x+dx,y,z) - f(w,x,y,z)) + abs(f(w,x-dx,y,z) - f(w,x,y,z)))
	sigy = 0.5*(abs(f(w,x,y+dy,z) - f(w,x,y,z)) + abs(f(w,x,y-dy,z) - f(w,x,y,z)))
	sigz = 0.5*(abs(f(w,x,y,z+dz) - f(w,x,y,z)) + abs(f(w,x,y,z-dz) - f(w,x,y,z)))
	sigval = np.sqrt(sigw**2 + sigx**2 + sigy**2 + sigz**2)
	return val, sigval

def FD_Err_vwxyz(f,v,dv,w,dw,x,dx,y,dy,z,dz):
	val = f(v,w,x,y,z)
	sigv = 0.5*(abs(f(v+dv,w,x,y,z) - f(v,w,x,y,z)) + abs(f(v-dv,w,x,y,z) - f(v,w,x,y,z)))
	sigw = 0.5*(abs(f(v,w+dw,x,y,z) - f(v,w,x,y,z)) + abs(f(v,w-dw,x,y,z) - f(v,w,x,y,z)))
	sigx = 0.5*(abs(f(v,w,x+dx,y,z) - f(v,w,x,y,z)) + abs(f(v,w,x-dx,y,z) - f(v,w,x,y,z)))
	sigy = 0.5*(abs(f(v,w,x,y+dy,z) - f(v,w,x,y,z)) + abs(f(v,w,x,y-dy,z) - f(v,w,x,y,z)))
	sigz = 0.5*(abs(f(v,w,x,y,z+dz) - f(v,w,x,y,z)) + abs(f(v,w,x,y,z-dz) - f(v,w,x,y,z)))
	sigval = np.sqrt(sigv**2 + sigw**2 + sigx**2 + sigy**2 + sigz**2)
	return val, sigval

def FD_Err_uvwxyz(f,u,du,v,dv,w,dw,x,dx,y,dy,z,dz):
	val = f(u,v,w,x,y,z)
	sigu = 0.5*(abs(f(u+du,v,w,x,y,z) - f(u,v,w,x,y,z)) + abs(f(u-du,v,w,x,y,z) - f(u,v,w,x,y,z)))
	sigv = 0.5*(abs(f(u,v+dv,w,x,y,z) - f(u,v,w,x,y,z)) + abs(f(u,v-dv,w,x,y,z) - f(u,v,w,x,y,z)))
	sigw = 0.5*(abs(f(u,v,w+dw,x,y,z) - f(u,v,w,x,y,z)) + abs(f(u,v,w-dw,x,y,z) - f(u,v,w,x,y,z)))
	sigx = 0.5*(abs(f(u,v,w,x+dx,y,z) - f(u,v,w,x,y,z)) + abs(f(u,v,w,x-dx,y,z) - f(u,v,w,x,y,z)))
	sigy = 0.5*(abs(f(u,v,w,x,y+dy,z) - f(u,v,w,x,y,z)) + abs(f(u,v,w,x,y-dy,z) - f(u,v,w,x,y,z)))
	sigz = 0.5*(abs(f(u,v,w,x,y,z+dz) - f(u,v,w,x,y,z)) + abs(f(u,v,w,x,y,z-dz) - f(u,v,w,x,y,z)))
	sigval = np.sqrt(sigu**2 + sigv**2 + sigw**2 + sigx**2 + sigy**2 + sigz**2)
	return val, sigval

def Stmu(x,dx):
	#Scatter to mean uncertainty
	xm = np.mean(x)
	stdx = np.sqrt((1/len(x))*np.sum((x-xm)**2))
	sigma = np.sqrt(stdx**2 + np.sum(dx**2))
	return xm, sigma

def Err(x,dx):
	if len(x) < 3: raise Exception("Use at least 3 points! ")
	dx = np.array([dx])
	N = len(x)
	if len(dx) == 1: dx = dx*np.ones(N)

	xm, sigx = Stmu(x,dx)
	sx = sigx*np.sqrt(N/(N-1))
	Sx = sx/np.sqrt(N)
	SSx = Sx/np.sqrt(N-2)
	return mx, sigx, sx, Sx, SSx

####### Fourier #######

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

def Fourier_Ex():
	print("L = np.pi\n",
	"N = 1\n", 
	"xbounds = np.array([0,L])\n",
	"a_0,a_n,b_n = Fourier_Const(f1,xbounds,N)\n",
	"x,y = Fourier_Eval(f1,xbounds,a_0,a_n,b_n,N)\n",
	"print(a_0,a_n,b_n)")

