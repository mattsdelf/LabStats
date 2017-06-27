#!usr/bin/env python3

"""
mypy

Matthew Del Favero's collection of useful python functions

Version 1.3

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
	return dy

def PDE_FD_1p_Err(x,sx,y,sy):
	x = np.asarray(x)
	y = np.asarray(y)
	if len(x) != len(y): raise Exception("Finite Difference: X and Y are not the same length!")
	dy = np.zeros_like(y)
	dyy = np.zeros_like(y)
	dyy[0] = sy[0]
	dyy[-1] = sy[-1]

	for i in range(len(x)-1):
		dy[i] = (y[i+1] - y[i])/(x[i+1] - x[i])
		dyy[i] = sy[i]*np.abs(dy[i]/y[i])
	return dy, dyy

def PDE_FD_2(x,y):
	x = np.asarray(x)
	y = np.asarray(y)
	if len(x) != len(y): raise Exception("Finite Difference: X and Y are not the same length!")
	ddy = np.zeros_like(y)
	for i in range(1,len(x)-1):
		ddy[i] = (y[i+1] + y[i-1] -2*y[i])/((x[i+1]-x[i])*(x[i]-x[i-1]))
	
	return ddy

def PDE_FD_2_Err(x,sx,y,sy):
	x = np.asarray(x)
	y = np.asarray(y)
	if len(x) != len(y): raise Exception("Finite Difference: X and Y are not the same length!")
	ddy = np.zeros_like(y)
	#ddyy = sy
	# The line above this one, which is also commented out seems to break the entire universe for some reason.
	ddyy = np.zeros_like(y)
	ddyy[0] = sy[0]
	ddyy[-1] = sy[-1]
	for i in range(1,len(x)-1):
		ddy[i] = (y[i+1] + y[i-1] -2*y[i])/((x[i+1]-x[i])*(x[i]-x[i-1]))
		ddyy[i] = sy[i]*np.abs(ddy[i]/y[i])
	return ddy, ddyy


######## Plotting ########

def Wire_Mesh_3D(x,y,z):
	fig = pl.figure()
	ax = fig.add_subplot(1,1,1,projections = '3d')
	X,Y = np.meshgrid(X,Y)
	ax.plot_wireframe(X,Y,Z)
	pl.show()

def Better_Limits(x,dx,y,dy,zero = "no"):
		
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
	return xlims, ylims
	

def Linear_Error_Textbox(xlims,ylims,m,sm,b,sb,rchi2,P):
	
	if sm >= m or sm == 0: mline = "m = %1.0e $\pm$ %1.0e\n"%(m,sm)
	elif m == 0: mline = "m = 0.00"
	else: 
		msigs = int(np.log(np.abs(m)) - np.log(np.abs(sm)))
		mline = "m = {:.{s}e} $\pm$ {:.{t}e}\n".format(m,sm,s = msigs,t=1)	
	if sb >= b or sb == 0: bline = "b = %1.0e $\pm$ %1.0e\n"%(b,sb)
	elif b == 0: bline = "b = 0.00"
	else: 
		bsigs = int(np.log(np.abs(b)) - np.log(np.abs(sb)))
		bline = "b = {:.{s}e} $\pm$ {:.{t}e}\n".format(b,sb,s = bsigs,t=1)	
	chiline = "$\chi^2$ = {:.{s}f}\n".format(rchi2,s = 2)
	Pline = "P = {:.{s}e}".format(P,s=3)
	Textbox = mline + bline + chiline + Pline

	return Textbox
	
def Gauss_Error_Textbox(xlims,ylims,a,sa,b,sb,c,sc,rchi2,P):
	
	if sa >= a or sa == 0: aline = "a = %1.0e $\pm$ %1.0e\n"%(a,sa)
	elif a == 0: aline = "a = 0.00\n"
	else: 
		asigs = int(np.log(np.abs(a)) - np.log(np.abs(sa)))
		aline = "a = {:.{s}e} $\pm$ {:.{t}e}\n".format(a,sa,s = asigs,t=1)	
	if sb >= b: bline = "b = %1.0e $\pm$ %1.0e\n"%(b,sb)
	elif b == 0 or sb == 0: bline = "b = 0.00\n"
	else: 
		bsigs = int(np.log(np.abs(b)) - np.log(np.abs(sb)))
		bline = "b = {:.{s}e} $\pm$ {:.{t}e}\n".format(b,sb,s = bsigs,t=1)	
	if sc >= c: cline = "c = %1.0e $\pm$ %1.0e\n"%(c,sc)
	elif c == 0 or sc == 0: cline = "c = 0.00\n"
	else: 
		csigs = int(np.log(np.abs(c)) - np.log(np.abs(sc)))
		cline = "c = {:.{s}e} $\pm$ {:.{t}e}\n".format(c,sc,s = csigs,t=1)	
	chiline = "$\chi^2$ = {:.{s}f}\n".format(rchi2,s = 2)
	Pline = "P = {:.{s}e}".format(P,s=3)
	Textbox = aline + bline + cline + chiline + Pline

	return Textbox
	
def Linear_Error_Fit_Plot(x,dx,y,dy, zero = "no" ,title = "No title", xword = "No X label", yword = "No Y label",text_coords = [0.3,0.75],savename = "no",show ="no"):

	if len(np.array([dx])) == 1: dx = dx*np.ones(len(x))
	if len(np.array([dy])) == 1: dy = dy*np.ones(len(y))
	m,sm,b,sb,rchi2,P = ODR_Linear_Fit(x,dx,y,dy)

	xlims, ylims = Better_Limits(x,dx,y,dy,zero)

	Y0 = m*xlims[0]+b
	Y1 = m*xlims[1]+b
	
	Textbox = Linear_Error_Textbox(xlims,ylims,m,sm,b,sb,rchi2,P)

	fig, ax = pl.subplots()
	ax.errorbar(x,y,xerr=dx,yerr=dy,fmt='ro',label = "Data")
	ax.plot(xlims,[Y0,Y1],label = "Fit")
	ax.set_title(title)
	ax.set_xlim(xlims)
	ax.set_ylim(ylims)
	ax.set_xlabel(xword)
	ax.set_ylabel(yword)
	ax.text(text_coords[0],text_coords[1],
		Textbox, fontsize = 17,
		horizontalalignment = 'center',
		verticalalignment = 'center',
		transform = ax.transAxes)
	if savename != "no": pl.savefig(savename)
	if show == "yes": pl.show()
	pl.close()
	
def Gauss_Error_Fit_Plot(x,dx,y,dy,param_guess, zero = "no" ,title = "No title", xword = "No X label", yword = "No Y label",text_coords = [0.3,0.75],savename = "no",show ="no"):

	#print("\n",x,"\n",dx,"\n",y,"\n",dy,"\n")
	if len(np.array([dx])) == 1: dx = dx*np.ones(len(x))
	if len(np.array([dy])) == 1: dy = dy*np.ones(len(y))
	a,sa,b,sb,c,sc,rchi2,P = ODR_Gauss_Fit(x,dx,y,dy,guess = param_guess)

	xlims, ylims = Better_Limits(x,dx,y,dy,zero)

	xplot = np.linspace(xlims[0],xlims[1],num=200)
	yplot = a*np.exp(-((xplot-b)/c)**2)
	
	Textbox = Gauss_Error_Textbox(xlims,ylims,a,sa,b,sb,c,sc,rchi2,P)

	fig, ax = pl.subplots()
	ax.errorbar(x,y,xerr=dx,yerr=dy,fmt='ro',label = "Data")
	ax.plot(xplot,yplot,label = "Fit")
	ax.set_title(title)
	ax.set_xlim(xlims)
	ax.set_ylim(ylims)
	ax.set_xlabel(xword)
	ax.set_ylabel(yword)
	ax.text(text_coords[0],text_coords[1],
		Textbox, fontsize = 17,
		horizontalalignment = 'center',
		verticalalignment = 'center',
		transform = ax.transAxes)
	if savename != "no": pl.savefig(savename)
	if show == "yes": pl.show()
	pl.close()


def Quick_Grid_Plot_To_File(grid,outfile = "redwop.png",color = "gray"):
	pl.figure()
	pl.imshow(grid,cmap=color)
	pl.savefig(outfile)
	pl.close()

def Quick_Hist(x, delimeter, outfile = "redwop.png", xword = "No X Label", yword = "No Y Label", title = "No Title", show = "no"):
	nbins = (np.max(x)-np.min(x))/delimeter
	pl.hist(x,nbins,facecolor='green',alpha=0.66)
	pl.xlabel(xword)
	pl.ylabel(yword)
	pl.title(title)
	pl.grid(True)
	pl.savefig(outfile)
	if show == "yes": pl.show()
	pl.close()

def PlotF(x,y,savename = "Bulbasaur.png", color = 'g', F1_Label = "Whale ", xword = "No X Label", yword = "No Y Label", title = "No Title", legend_loc = "upper right",linsty = '-' ,show = "no"):
	pl.plot(x,y, color, linestyle = linsty, label = F1_Label)
	pl.xlabel(xword)
	pl.ylabel(yword)
	pl.title(title)
	pl.legend(loc = legend_loc)
	pl.savefig(savename)
	if show == "yes": pl.show()
	pl.close()

def PlotF2(x,y1,y2,savename = "Ivysaur.png", color1 = 'g', color2 = 'b', F1_Label = "Whale ", F2_Label = "Baluga", xword = "No X Label", yword = "No Y Label", title = "No Title", legend_loc = "upper right",linsty1 = '-' , linsty2 = '--', show = "no"):
	pl.plot(x,y1, color1, linestyle = linsty1, label = F1_Label)
	pl.plot(x,y2, color2, linestyle = linsty2, label = F2_Label)
	pl.xlabel(xword)
	pl.ylabel(yword)
	pl.title(title)
	pl.legend(loc = legend_loc)
	pl.savefig(savename)
	if show == "yes": pl.show()
	pl.close()

def PlotF3(x,y1,y2,y3,savename = "Venusaur.png", color1 = 'g', color2 = 'b', color3 = 'r', F1_Label = "Whale ", F2_Label = "Baluga", F3_Label = "Blue", xword = "No X Label", yword = "No Y Label", title = "No Title", legend_loc = "upper right",linsty1 = '-' , linsty2 = '--', linsty3 = ':',show = "no"):
	pl.plot(x,y1, color1, linestyle = linsty1, label = F1_Label)
	pl.plot(x,y2, color2, linestyle = linsty2, label = F2_Label)
	pl.plot(x,y3, color3, linestyle = linsty3, label = F3_Label)
	pl.xlabel(xword)
	pl.ylabel(yword)
	pl.title(title)
	pl.legend(loc = legend_loc)
	pl.savefig(savename)
	if show == "yes": pl.show()
	pl.close()



######## Error Management ########

def rms(A): return np.sqrt(np.mean(np.square(A)))

def std(A): return np.sqrt(np.sum((A-np.mean(A))**2))

def ODR_Linear_Fit_f(B,x): return B[0]*x + B[1]

def ODR_Gauss_Fit_f(B,x): return B[0]*np.exp(-0.5*((x-B[1])/B[2])**2)

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
	#A = myoutput.pprint()

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
	#myoutput.pprint()
	
	m = myoutput.beta[0]
	b = myoutput.beta[1]
	sm = myoutput.sd_beta[0] 
	sb = myoutput.sd_beta[1] 
	rchi2 = myoutput.res_var
	dof = N - len(myoutput.beta)
	chi2 = rchi2*dof
	P = 1 - scichi.cdf(chi2,dof)
	return m, sm, b, sb, rchi2, P

def ODR_Gauss_Fit(X,dX,Y,dY,guess):
	#print(X,dX,Y,dY,guess)

	if len(X) != len(Y): raise Exception("The vectors are not the same length! ")
	if len(X) < 3: raise Exception("Use at least 3 points! ")
	dX = np.array([dX])
	dY = np.array([dY])
	
	N = len(X)
		
	if len(dX) == 1: dX = dX*np.ones(N)
	if len(dY) == 1: dY = dY*np.ones(N)

	Gauss = odr.Model(ODR_Gauss_Fit_f)
	mydata = odr.RealData(X,y=Y,sx=dX,sy=dY)
	myodr = odr.ODR(mydata,Gauss,beta0 = guess,maxit=1000)
	myoutput = myodr.run()
	#myoutput.pprint()
	
	a = myoutput.beta[0]
	b = myoutput.beta[1]
	c = myoutput.beta[2]
	sa = myoutput.sd_beta[0] 
	sb = myoutput.sd_beta[1] 
	sc = myoutput.sd_beta[2] 
	rchi2 = myoutput.res_var
	dof = N - len(myoutput.beta)
	chi2 = rchi2*dof
	P = 1 - scichi.cdf(chi2,dof)
	return a, sa, b, sb, c, sc, rchi2, P

def Weighted_Mean(x,sx):
	xm = np.sum(x/(sx**2))/np.sum(1/(sx**2))
	sxm = np.sqrt(1/np.sum(1/(sx**2)))
	return xm, sxm

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

