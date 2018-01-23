#!usr/bin/env python3

"""
LabStats

Matthew Del Favero's collection of Modern Lab python functions for python 3

See README

Version 1.3

"""

######## Imports ########

import math
import numpy as np
import pylab as pl

# These packages are used for uncertainty calculations and model fitting.
from scipy.stats import chi2 as scichi
import scipy.odr as odr

######## Misc ########

# Error by sum of squares.
# Note that this is not the standard deviation because it is not divided by the root of the length of the set.
def std(A): return np.sqrt(np.sum((A-np.mean(A))**2))

# Root-Mean-Square calculations
def rms(A): return np.sqrt(np.mean(np.square(A)))

# Given a set of data-points, create limits for the graph such that the error-bars do not go off the side of the graph.
def Better_Limits(x,dx,y,dy,zero = "no"):
	# Create a set of limits based on the minimums and maximums of the data.	
	xlims = np.array([np.min(x),np.max(x)])
	ylims = np.array([np.min(y),np.max(y)])
	# Calculate the length of each axis
	Lx = np.max(x) - np.min(x)
	Ly = np.max(y) - np.min(y)
	# Expand those limits by the error from the point with the largest error, and also by a tenth of the length of the previous axis.
	xlims[0] -= 0.1*Lx + np.max(dx)
	xlims[1] += 0.1*Lx + np.max(dx)
	ylims[0] -= 0.1*Ly + np.max(dy)
	ylims[1] += 0.1*Ly + np.max(dy)
	# If you want to "zero" the graph, forcing the focus of the plot into the first quadrant, set the lower limit below zero by the maximum error.
	if zero == "yes":
		xlims[0] = -np.max(dx)
		ylims[0] = -np.max(dy)
	return xlims, ylims

# Calculate the weighted mean of a set of data, and the uncertainty therein.
def Weighted_Mean(x,sx):
	xm = np.sum(x/(sx**2))/np.sum(1/(sx**2))
	sxm = np.sqrt(1/np.sum(1/(sx**2)))
	return xm, sxm

######## Models for ODR ########

# A constant model
def ODR_Constant_f(B,x): return B[0]

# A linear model where the intercept is forced to be zero
def ODR_Slope_f(B,x): return B[0]*x

# A linear model
def ODR_Linear_f(B,x): return B[0]*x + B[1]

# A power fit model
#def ODR_Power_f(B,x): return B[1]*x**B[0] + B[2]

# An exponential model
def ODR_Exponential_f(B,x): return B[0]*np.exp((x-B[1])/B[2])

# An model for decaying exponential growth
def ODR_Decaying_Growth_f(B,x): return B[0]*(1 - np.exp((x-B[1])/B[2]))

# A gaussian model
def ODR_Gauss_f(B,x): return B[0]*np.exp(-0.5*((x-B[1])/B[2])**2)

# A gaussian plus a line
def ODR_Gaussline_f(B,x): return B[0]*np.exp(-0.5*((x-B[1])/B[2])**2) + B[3]*x + B[4]

#### The Rest of the models in this section are for the Double-Slit experiment####
# Note that you will have to adjust parameters in these functions manually.
# I have not found an easy work-around to this for two reasons.
# The first is that I don't know if there is an easy way to introduce a constant into a proper ODR "model"
# The second is my whole nested organization of functions and modular inputs.

# A Single Slit model
def ODR_Single_Slit_f(B,x): 
	# You will need to change this manually
	Lam = 656*10**(-9)
	alpha = (B[0]*np.pi/Lam)*np.sin(x)
	y = (np.sin(alpha)/alpha)**2
	return y

# A Double Slit model
def ODR_Double_Slit_f(B,x): 
	# You will need to change this manually
	Lam = 656*10**(-9)
	alpha = (B[0]*np.pi/Lam)*np.sin(x)
	beta = (B[1]*np.pi/Lam)*np.sin(x)
	y = (np.sin(alpha)/alpha)**2 * (np.cos(beta))**2
	return y

# A model for two different slits
def ODR_Two_Slits_f(B,x):
	# You will need to change this manually
	Lam = 656*10**(-9)
	alpha1 = (B[0]*np.pi/Lam)*np.sin(x)
	alpha2 = (B[1]*np.pi/Lam)*np.sin(x)
	beta = (B[2]*np.pi/Lam)*np.sin(x)
	y = (1/(4*B[0]*B[1]))*(2*B[0]*B[1]*(np.cos(2*beta))*(np.sin(alpha1)/alpha1)*(np.sin(alpha2)/alpha2) + (B[0]**2)*(np.sin(alpha1)/alpha1)**2 + (B[1]**2)*(np.sin(alpha2)/alpha2)**2)
	return y

# A model for a single slit with a convolution
def ODR_Single_SlitC_f(B,x):
	# You will need to change this manually
	Lam = 656*10**(-9)
	# You will need to change this manually
	TopHat = 3

	# define the function for your convolution to be the corresponding function without the convolution
	y = ODR_Single_Slit_f(B,x)
	# Create the array to hold the convolution
	Envelope = np.zeros_like(x)
	# Find the approximate center point
	x0 = int(len(x)/2)
	# Special case for TopHat = 1. By the way, using a tophat function which has a width of one SHOULDN'T be any different from not using the convolution.
	if TopHat == 1: Envelope[x0] = 1
	# Fill the envelope with the appropriate number of ones.
	else: 
		for i in range(int(x0-(TopHat/2)),int(x0+(TopHat/2))): Envelope[i] = 1
	
	# Run the np.convolve convolution
	Conv = (np.convolve(y,Envelope,mode = 'same')/TopHat)
	return Conv

# A model for a double slit with a convolution
def ODR_Double_SlitC_f(B,x):
	# You will need to change this manually
	Lam = 656*10**(-9)
	# You will need to change this manually
	TopHat = 3

	# define the function for your convolution to be the corresponding function without the convolution
	y = ODR_Double_Slit_f(B,x)
	# Create the array to hold the convolution
	Envelope = np.zeros_like(x)
	# Find the approximate center point
	x0 = int(len(x)/2)
	# Special case for TopHat = 1. By the way, using a tophat function which has a width of one SHOULDN'T be any different from not using the convolution.
	if TopHat == 1: Envelope[x0] = 1
	# Fill the envelope with the appropriate number of ones.
	else: 
		for i in range(int(x0-(TopHat/2)),int(x0+(TopHat/2))): Envelope[i] = 1
	
	# Run the np.convolve convolution
	Conv = (np.convolve(y,Envelope,mode = 'same')/TopHat)
	return Conv

# A model for two different slits with a convolution
def ODR_Two_SlitsC_f(B,x):
	# You will need to change this manually
	Lam = 656*10**(-9)
	# You will need to change this manually
	TopHat = 3

	# define the function for your convolution to be the corresponding function without the convolution
	y = ODR_Two_Slits_f(B,x)
	# Create the array to hold the convolution
	Envelope = np.zeros_like(x)
	# Find the approximate center point
	x0 = int(len(x)/2)
	# Special case for TopHat = 1. By the way, using a tophat function which has a width of one SHOULDN'T be any different from not using the convolution.
	if TopHat == 1: Envelope[x0] = 1
	# Fill the envelope with the appropriate number of ones.
	else: 
		for i in range(int(x0-(TopHat/2)),int(x0+(TopHat/2))): Envelope[i] = 1
	
	# Run the np.convolve convolution
	Conv = (np.convolve(y,Envelope,mode = 'same')/TopHat)
	return Conv


######## Utilities ########

# Create the textbox for the plot.
#
# This is just a string with a few line-breaks in it.
#
# I discovered that there is a limited "math mode" for exponents and simple commands.
# However, it does not have the full functionalities of the LaTeX math mode.
#
# This function creates a line for every parameter in the fit with their associated uncertainty.
# It creates a line for the Chi-squared value and a line for the P value
def Textbox(fit,B,sB,rchi2,P):
	# Declare an empty string to add other string to.
	params = ""
	# For loop for each parameter
	for i in range(len(B)):
		# If the error comes through as zero, there is a bug elsewhere in the code, or your guess is bad.
		# I handle the case where the error in a parameter is greater than the value of the parameter by giving the measurement one sig fig.
		# POSSIBLE BUG: Why isn't this the absolute value of the parameter?
		# In this line, I use the traditional method for formated strings.
		if sB[i] >= B[i] or sB[i] == 0: Iline = "B%d = %1.0e $\pm$ %1.0e\n"%(i,B[i],sB[i])
		# If the value of the parameter itself is zero, then it probably doesn't belong in the model, or it's a bad guess.
		# In either case, a parameter with a value of zero will bother the logarythmic math below.
		# Here I begin to use the python3 formatted string notation.
		elif B[i] == 0: Iline = "B{:} = 0.00\n".format(i)
		# Else, calculate the appropriate number of sigfigs by comparing the log of the value of the parameter with the log of the error of the parameter.
		# Here, I needed to use the python3 formatted string notation because it allowed me to do some spooky magic with the sig figs.
		else: 
			Isigs = int(np.log10(np.abs(B[i])) - np.log10(np.abs(sB[i]))) +1
			Iline = "B{:} = {:.{s}e} $\pm$ {:.{t}e}\n".format(i,B[i],sB[i],s = Isigs, t = 1)
		# Add the line for a parameter to the string for all the parameters.
		params += Iline
	
	# Here are the far less complicated lines that simply dislpay the chi square value and p vale
	chiline = "$\chi^2$ = {:.{s}f}\n".format(rchi2,s = 2)
	Pline = "P = {:.{s}e}".format(P,s=3)

	# Create the string for the whole textbox.
	Textbox = params + chiline + Pline
	return Textbox

# This function takes the string for the name of the fit, and returns the function, evaluated.
def Fit_Plot(x,B,fit):
	if fit == "constant": return ODR_Constant_f(B,x)
	elif fit == "slope": return ODR_Slope_f(B,x)
	elif fit == "linear": return ODR_Linear_f(B,x)
	#elif fit == "power": return ODR_Power_f(B,x)
	elif fit == "exponential": return ODR_Exponential_f(B,x)
	elif fit == "decay_growth": return ODR_Decaying_Growth_f(B,x)
	elif fit == "gauss": return ODR_Gauss_f(B,x)
	elif fit == "gaussline": return ODR_Gaussline_f(B,x)
	elif fit == "single_slit": return ODR_Single_Slit_f(B,x)
	elif fit == "double_slit": return ODR_Double_Slit_f(B,x)
	elif fit == "two_slits": return ODR_Two_Slits_f(B,x)
	elif fit == "single_slit_c": return ODR_Single_SlitC_f(B,x)
	elif fit == "double_slit_c": return ODR_Double_SlitC_f(B,x)
	elif fit == "two_slits_c": return ODR_Two_SlitsC_f(B,x)
	else: raise Exception("I can't plot that!")
	return 1

# This is where the "fitting" occurs.
# Make sure that your guess is reasonable, and as close to your values as possible.
# This function takes the data, a guess, and the name of the fit.
# This function returns the output parameters of the fit
def ODR_Fit(X,dX,Y,dY,guess,fit):

	if fit == "constant": mymodel = odr.Model(ODR_Constant_f)
	elif fit == "slope": mymodel = odr.Model(ODR_Slope_f)
	elif fit == "linear": mymodel = odr.Model(ODR_Linear_f)
	#elif fit == "power": mymodel = odr.Model( ODR_Power_f)
	elif fit == "exponential": mymodel = odr.Model(ODR_Exponential_f)
	elif fit == "decay_growth": mymodel = odr.Model(ODR_Decaying_Growth_f)
	elif fit == "gauss": mymodel = odr.Model(ODR_Gauss_f)
	elif fit == "gaussline": mymodel = odr.Model(ODR_Gaussline_f)
	elif fit == "single_slit": mymodel = odr.Model(ODR_Single_Slit_f)
	elif fit == "double_slit": mymodel = odr.Model(ODR_Double_Slit_f)
	elif fit == "two_slits": mymodel = odr.Model(ODR_Two_Slits_f)
	elif fit == "single_slit_c": mymodel = odr.Model(ODR_Single_SlitC_f)
	elif fit == "double_slit_c": mymodel = odr.Model(ODR_Double_SlitC_f)
	elif fit == "two_slits_c": mymodel = odr.Model(ODR_Two_SlitsC_f)
	else: raise Exception("Unknown Function")

	mydata = odr.RealData(X,y=Y,sx=dX,sy=dY)
	myodr = odr.ODR(mydata,mymodel,beta0 = guess,maxit = 1000)
	myoutput = myodr.run()
	#myoutput.pprint()
	
	Beta = myoutput.beta[:]
	sBeta = myoutput.sd_beta[:]
	rchi2 = myoutput.res_var
	N = len(X)
	dof = N - len(myoutput.beta)
	chi2 = rchi2*dof
	P = 1 - scichi.cdf(chi2,dof)
	return Beta, sBeta, rchi2, P


######## Lab ODR ########

# This is the 'master-function'
# For usage, see the usage section.
def Lab_ODR(x,dx,y,dy,fit, param_guess, zero = "no", text_coords = [0.3,0.75],savename = "no",show ="no",output = "no", **kwargs):

	if len(x) != len(y): raise Exception("The vectors are not the same length! ")
	if len(x) < 3: raise Exception("Use at least 3 points! ")

	x = np.asarray(x)
	dx = np.asarray([dx])
	y = np.asarray(y)
	dy = np.asarray([dy])

	if len(dx) == 1: dx = dx[0]*np.ones(len(x))
	if len(dy) == 1: dy = dy[0]*np.ones(len(y))
	B, sB, rchi2, P = ODR_Fit(x,dx,y,dy,param_guess,fit)

	if show != "no" or savename != "no":
		
		xlims, ylims = Better_Limits(x,dx,y,dy,zero)
		xplot = np.linspace(xlims[0],xlims[1],num = 2000)
		yplot = Fit_Plot(xplot,B,fit)

		Text = Textbox(fit,B,sB,rchi2,P)
		fig, ax = pl.subplots()
		ax.errorbar(x,y,xerr=dx,yerr=dy,fmt='ro',label = "Data")
		ax.plot(xplot,yplot,label = "Fit")
		ax.set_xlim(xlims)
		ax.set_ylim(ylims)
		ax.text(text_coords[0],text_coords[1],
			Text, fontsize = 17,
			horizontalalignment = 'center',
			verticalalignment = 'center',
			transform = ax.transAxes)
		ax.set(**kwargs)
		if show != "no": pl.show()
		if savename != "no": pl.savefig(savename)
		pl.close()

	if output == "full": return B, sB, rchi2, P
	if output == "B": return B
	return


######## Finite Difference Stuff ########
# Below is the finite difference method for approximating error.
# All of the functions in this section serve the same purpose, but for different numbers of variables and constants.

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

######## Misc Error Stuff ########

# This function just calculates the man and standard deviation of the data given.
def Stmu(x,dx):
	xm = np.mean(x)
	stdx = np.sqrt((1/len(x))*np.sum((x-xm)**2))
	sigma = np.sqrt(stdx**2 + np.sum(dx**2))
	return xm, sigma

# This function calculates the mean, standard deviation, standard error, the error in the standard error, and the error in that error, and returns those.
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

