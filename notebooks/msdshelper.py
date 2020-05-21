import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sympy import Symbol, symbols, integrate
from scipy.integrate import odeint
from scipy import integrate, optimize

tab10 = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', 
         '#8C564B', '#CFECF9', '#7F7F7F', '#BCBD22', '#17BECF']  ## Define color table for figures

def exponential(x,a,b):
    return a * np.exp(b*x)

def logistic(x,ymax,A,r):
    return ymax/(1+ A*np.exp(-r*(x)))


def growth_cases_deaths(ycases, ydeaths, country, semilog=True):
	fig, ax1 = plt.subplots(figsize=(10,6));
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	lag_days = 0

	x = np.arange(0, len(ycases))
	popt, pcov = curve_fit(exponential, x, ycases, p0 = [0, 0])

	x_fit = np.arange(0, len(ycases))
	y_exp_fit = exponential(x_fit, *popt)

	rate=(np.exp(tuple(popt)[1]))-1
	double= np.log(2)/np.log(1+ rate)
	R_squared=r2_score(y_exp_fit, ycases[-len(y_exp_fit):])


	ax1.plot(x-lag_days,ycases,'o', ms=10, markerfacecolor='none', color = tab10[0], 
	              label='USA Cases double every %.2f days with $R^2$ = %.2f delayed ' %(double, R_squared));
	ax1.plot(x_fit-lag_days, y_exp_fit,'--', lw=2, color = tab10[0])

	lag_days=np.empty(len(x))
	lag_days.fill(0)   ### Lagging is another fitting parameter, 4 serves this fit well, explanation in another time

	Normalized=np.max(ycases)/np.max(ydeaths)
	#Normalized =1 # actual data
	x = np.arange(0, len(ydeaths))
	popt, pcov = curve_fit(exponential, x, ydeaths, p0 = [0, 0])


	x_fit = np.arange(0, len(ydeaths))
	y_exp_fit = exponential(x_fit, *popt)


	rate=(np.exp(tuple(popt)[1]))-1
	double= np.log(2)/np.log(1+ rate)
	R_squared=r2_score(y_exp_fit, ydeaths[-len(y_exp_fit):])

	
	

	ax2.plot(x-lag_days, ydeaths ,'o', ms=10, markerfacecolor=tab10[1], color = tab10[1], 
              label='USA Deaths doubles every %.2f days \n with $R^2$ = %.2f delayed by %.2f days Normalized by %.2f' %(double, R_squared, lag_days[0], Normalized));
           #  label='USA Deaths, Normalized by %.2f' % Normalized);
	ax2.plot(x_fit-lag_days, y_exp_fit,'--', lw=2, \
		color = tab10[1])

	if semilog == True:
		ax1.set_yscale('log')
		ax2.set_yscale('log')

	plt.legend(frameon=False, fontsize=10)
	plt.xlim(-1, len(ydeaths)+1)
	ax1.set_ylim(0.9, 20000)
	ax2.set_ylim(0.9, 1000)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	ax1.set_xlabel('Days after the Onset of Significant Number of Infected Patients', size=12);
	ax1.set_ylabel('Total Infected Patients - open circles', size=12, color=tab10[0]);
	ax1.tick_params(axis='y', labelcolor=tab10[0])
	ax2.set_ylabel('Deaths (Normalized & Shifted) - filled circles', size=12, color=tab10[1]);
	ax2.tick_params(axis='y', labelcolor=tab10[1])

	return


def growth_cases_exp_countries(df, countrylist, semilog=True):
	plt.figure(figsize=(10,6));
	lag_days = 0
	
	icolor = 0
	for country in countrylist:
		y = df[df.Country == country]['Cases_Cumulative']
		x = np.arange(0, len(y))

		popt, pcov = curve_fit(exponential, x, y, p0 = [0, 0])
		x_fit = np.arange(0, len(y))
		y_exp_fit = exponential(x_fit, *popt)

		rate=(np.exp(tuple(popt)[1]))-1
		double= np.log(2)/np.log(1+ rate)
		R_squared=r2_score(y_exp_fit, y[-len(y_exp_fit):])

		if semilog == True:
			plt.semilogy(x-lag_days,y,'o', ms=10, markerfacecolor='none', color = tab10[icolor], 
		              label= country + ' Cases doubles every %.2f days with $R^2$ = %.2f delayed ' %(double, R_squared));
		           #  label='USA Deaths, Normalized by %.2f' % Normalized);
			plt.semilogy(x_fit-lag_days, y_exp_fit,'--', lw=2.5, color = tab10[icolor])
		else:
			plt.plot(x-lag_days,y,'o', ms=10, markerfacecolor='none', color = tab10[icolor], 
		              label=country + ' Cases doubles every %.2f days with $R^2$ = %.2f delayed ' %(double, R_squared));
		           #  label='USA Deaths, Normalized by %.2f' % Normalized);
			plt.plot(x_fit-lag_days, y_exp_fit,'--', lw=2.5, color = tab10[icolor])
		icolor = icolor + 1

		
	plt.legend(frameon=False, fontsize=10)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	plt.xlim(-1, len(y)+1)

	plt.xlabel('Days after the Onset of Significant Number of Infected Patients', size=12);
	plt.ylabel('Total (Confirmed) Infected Patients', size=12);

	return 

def growth_cases_deaths_logi(ycases, ydeaths, country, ymax = None, lagdays = 0, semilog=True):
	fig, ax1 = plt.subplots(figsize=(10,8));

	x = np.arange(0, len(ycases))
	popt, pcov = curve_fit(exponential, x, ycases, p0 = [0, 0])
	popt2, pcov2 = curve_fit(logistic, x, ycases, p0 = [10000,1,.3])

	x_fit = np.arange(0, len(ycases))
	y_exp_fit = exponential(x_fit, *popt)
	y_logi_fit = logistic(x_fit, *popt2)

	x_projection = np.arange(len(ycases), len(ycases)+14)
	y_projection=logistic(x_projection, *popt2)
	R_squared=r2_score(y_logi_fit, ycases[-len(y_logi_fit):])

	ax1.plot(x,ycases,'o', ms=8, markerfacecolor='none', color = tab10[0],
	         label='Infected Cases');
	ax1.plot(x_fit, y_logi_fit,'-', lw=2, color = tab10[0], 
	         label = 'Logistic Fit: Asymptotic decline with $R^2$= %.2f' %R_squared)
	ax1.set_ylabel('Number of Confirmed Cases (Infected)', color=tab10[0]) 
	ax1.tick_params(axis='y', labelcolor=tab10[0])
	ax1.set_xlabel('Days after the Onset of Significant Number of Infected Patients')
	ax1.set_xlim(0, 30);
	
	Normalized=np.max(ycases)/np.max(ydeaths)
	ydeaths = ydeaths * Normalized
	#Normalized =1 # actual data
	x = np.arange(0, len(ydeaths))

	lag_days=np.empty(len(x))
	lag_days.fill(lagdays)   ### Lagging is another fitting parameter, 4 serves this fit well, explanation in another time

	popt, pcov = curve_fit(exponential, x, ydeaths, p0 = [0, 0])
	popt2, pcov2 = curve_fit(logistic, x, ydeaths, p0 = [10000,1,.3])

	x_fit = np.arange(0, len(ydeaths))
	y_exp_fit = exponential(x_fit, *popt)
	y_logi_fit = logistic(x_fit, *popt2)

	x_projection = np.arange(len(ydeaths), len(ydeaths)+14)
	y_projection=logistic(x_projection, *popt2)
	R_squared=r2_score(y_logi_fit, ydeaths[-len(y_logi_fit):])

	ax2 = ax1.twinx()

	ax2.plot(x-lag_days, ydeaths,'o', ms=8, markerfacecolor= tab10[1], 
         color = tab10[1],
         label='Deaths (normalized)');
	ax2.plot(x_fit-lag_days, y_logi_fit,'-', lw=2, color = tab10[1],
	        label = 'Logistic fit: with $R^2$= %.2f \ndelayed by %.2f days normalized by %.2f' % (R_squared, lag_days[0], Normalized))
	ax2.set_ylabel('Deaths (Normalized & Shifted)', color=tab10[1]) 
	ax2.tick_params(axis='y', labelcolor=tab10[1])

	if ymax != None:
		ax2.set_ylim(0.9, ymax);
		ax1.set_ylim(0.9, ymax);

	ax1.legend(frameon=False, loc=2, fontsize=10);
	ax2.legend(frameon=False, loc=1, fontsize=10);
	fig.tight_layout();

	if semilog == True:
		ax1.set_yscale('log')
		ax2.set_yscale('log')

	plt.suptitle('Growth Trends for Confirmed Cases and Deaths in ' + country + ' (COVID-19)', y = 1.05);
	plt.xlim(0, len(x)-lag_days[0] +3);
	plt.gca().spines['top'].set_visible(False);
	plt.gca().spines['right'].set_visible(False);

	return

def running_exponential(df, countries, ymax = None, xmax = None):
	plt.figure(figsize = (13,9))
	range_i = 11

	icolor = 0
	for country in countries:
		double_country = [] 
		R_squared_country = []
		y = df[country]

		for i in range(0, len(y)-range_i + 2):
			x = np.arange(i, len(y)- range_i + i)
			yy = y[i: len(y) - range_i + i]
			popt, pcov = curve_fit(exponential, x, yy, p0 = [0,0])

			x_fit = np.arange(i, len(y)-range_i + i)
			y_exp_fit = exponential(x_fit, *popt)
			rate=(np.exp(tuple(popt)[1]))-1
			double= np.log(2)/np.log(1+ rate)
			R_squared=r2_score(y_exp_fit, yy[-len(y_exp_fit):])

			double_country = np.append(double_country, double)
			R_squared_country = np.append(R_squared_country, R_squared)

		x = np.arange(len(y)-range_i, len(y)+1)
		plt.plot(x,double_country,'o--', ms=10, color = tab10[icolor], label=country)
		error=double_country*(1-R_squared_country)
		plt.errorbar(x,double_country, yerr=error, uplims=True, lolims=True,
		             label='', color = tab10[icolor]) 
		
		icolor = icolor + 1
		a=plt.gca() 

	if ymax != None:
		plt.ylim(0, ymax)
	else:
		plt.ylim(0.5, 45)

	if xmax != None:
		plt.xlim(10, xmax)
	else:
		plt.xlim(10, 25)

	plt.legend(frameon=False)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	plt.xlabel('March 2020', size=15);
	plt.ylabel('Number of Days to Double the Number of Infected People', size=15);

	return

# The differential equations for the SIR model 
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# The Integration procedure
def fit_odeint(y, beta, gamma):
    return integrate.odeint(sir_model, (S0, I0, R0), t, args=(N, beta, gamma))[:,1]

# The differential equations for the SIR model 
def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - (sigma * E)
    dIdt = (sigma * E) - (gamma * I)
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def SEIR_curves(beta, gamma, N, sigma, t, infected_only = False):

	I0 = 4 #Initial Infected
	R0 = 0.0 #Initial Recovered
	E0=0 #Initial Exposed but not infected
	S0 = N - I0 #Initial Susceptible

	# Initial conditions vector
	y0 = S0, E0, I0, R0
	# Integrate the SIR equations over the time grid, t.
	ret = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
	S, E, I, R = ret.T

	# Plot the data on three separate curves for S(t), I(t) and R(t)
	plt.rcParams.update({'font.size': 18})

	plt.figure(figsize=(15,10))
	
	if infected_only == True:
		plt.plot(t, I, color = tab10[2], alpha=0.75, lw=4, label='Infected')
	else:
		plt.plot(t, S, color = tab10[0], alpha=0.75, lw=4, label='Susceptible')
		plt.plot(t, E, color = tab10[1], alpha=0.75, lw=4, label='Exposed')
		plt.plot(t, I, color = tab10[2], alpha=0.75, lw=4, label='Infected')
		plt.plot(t, R, color = tab10[3], alpha=0.75, lw=4, label='Recovered with immunity')
		plt.legend(frameon = False)

	plt.xlabel('Time (in days)')
	plt.ylabel('Infected Cases')
	#ax.set_ylim(0,1.2)
	plt.gca().yaxis.set_tick_params(length=0)
	plt.gca().xaxis.set_tick_params(length=0)
	# plt.gca().grid(b=True, which='major', c='lightgray', lw=2, ls='-')
	# legend.get_frame().set_alpha(0.5)
	for spine in ('top', 'right'):
	    plt.gca().spines[spine].set_visible(False)

	return

def SIR_optimize(i_daily, i_cumulative, country, init_pop):
	#DEFINE PARAMETERS and INITIAL CONDITIONS here
	global beta, gamma, N, I0, R0, S0, t

	t = np.arange(0, len(i_daily))
	beta = 0.5  # beta = Contact rate
	gamma = 0.10 # gamma = mean recovery rate (in 1/days)
	N = init_pop       # Initial Susceptible population

	I0 = 4 #Initial Infected
	R0 = 0.0 #Initial Recovered
	S0 = N - I0 #Initial Susceptible

	#popt and pcov stores beta and gamma values/error variance
	popt, pcov = optimize.curve_fit(fit_odeint, t, i_daily) #Optimization procedure

	fitted = fit_odeint(i_daily, *popt) #fitted projection
	fitted_cum=np.cumsum(fitted)  #cumulative sum of the projection

	#PLOTTING THE RESULTS

	fig, ax1 = plt.subplots(figsize=(12,8))
	
	#Infected Cases - per Day
	plt.rcParams.update({'font.size': 18})
	ax1.set_xlabel('time (in days)', fontsize=15)
	ax1.set_ylabel('cases per day', color=tab10[1], fontsize=15)
	ax1.plot(t, i_daily,'o', color=tab10[1], alpha = 0.75,)
	ax1.plot(t, fitted, lw = 4, alpha = 0.75, color=tab10[1])
	ax1.tick_params(axis='y', labelcolor=tab10[1])

	# Infected Cases - Cumulative
	ax2 = ax1.twinx()  
	ax2.set_ylabel('cumulative cases', color=tab10[0], fontsize=15)  # we already handled the x-label with ax1
	ax2.plot(t, i_cumulative,'o', color=tab10[0], alpha = 0.75,)
	ax2.plot(t, fitted_cum, lw = 4, alpha = 0.75, color=tab10[0])
	ax2.tick_params(axis='y', labelcolor=tab10[0])

	return popt[0],popt[1],popt[0]/popt[1]

def SIR_fit_param(i_daily, i_cumulative, init_pop, country, b = 0.5, g = 0.10, r_naught = 3.34):
	#DEFINE PARAMETERS and INITIAL CONDITIONS here
	global beta, gamma, N, I0, R0, S0, t

	t = np.linspace(0, 40, 40)
	beta = b  		# beta = Contact rate
	gamma = g 		# gamma = mean recovery rate (in 1/days)
	N = init_pop        # Initial Susceptible population

	I0 = 4 #Initial Infected
	R0 = 0.0 #Initial Recovered
	S0 = N - I0 #Initial Susceptible

	y0 = S0, I0, R0
	ret = odeint(sir_model, y0, t, args=(N, beta, gamma))
	S, I, R = ret.T

	# Plot the data on three separate curves for S(t), I(t) and R(t)
	plt.rcParams.update({'font.size': 15})

	fig = plt.figure(figsize=(12,8))
	# ax = fig.add_subplot(111, axisbelow=True)
	# #ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
	# #ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

	x = np.arange(0, len(i_daily))

	plt.plot(t, I, linestyle = '-', lw=4, alpha = 0.75, label='SIR Model - daily infected', color=tab10[0])
	plt.plot(t, np.cumsum(I), linestyle = '-', lw=4, alpha = 0.75, label='SIR Model - total confirmed cases', color = tab10[1])
	plt.plot(x, i_daily, 'o', label= country + ' daily (new) infected cases', color = tab10[0])
	plt.plot(x, i_cumulative, 'o', label= country + ' total confirmed cases', color = tab10[1])

	plt.xlabel('Time (in days)')
	plt.ylabel('Infected Cases')
	#ax.set_ylim(0,1.2)
	plt.gca().yaxis.set_tick_params(length=0)
	plt.gca().xaxis.set_tick_params(length=0)
	# ax.grid(b=True, which='major', c='w', lw=2, ls='-')
	plt.legend(frameon=False, fontsize=12)
	for spine in ('top', 'right'):
	    plt.gca().spines[spine].set_visible(False)

	return

def SIR_curves(i_daily, i_cumulative, init_pop, country, b = 0.5, g = 0.10, r_naught = 3.34):
	#DEFINE PARAMETERS and INITIAL CONDITIONS here
	global beta, gamma, N, I0, R0, S0, t

	t = np.linspace(0, 40, 40)
	beta = b  		# beta = Contact rate
	gamma = g 		# gamma = mean recovery rate (in 1/days)
	N = init_pop        # Initial Susceptible population

	I0 = 4 #Initial Infected
	R0 = 0.0 #Initial Recovered
	S0 = N - I0 #Initial Susceptible

	y0 = S0, I0, R0
	ret = odeint(sir_model, y0, t, args=(N, beta, gamma))
	S, I, R = ret.T

	# Plot the data on three separate curves for S(t), I(t) and R(t)
	plt.rcParams.update({'font.size': 18})

	fig = plt.figure(figsize=(15,10))
	ax = fig.add_subplot(111, axisbelow=True)
	
	i_cum_fit = np.cumsum(I)
	x = np.arange(0, len(i_daily))

	ax.plot(t, S, color = tab10[0], alpha=0.75, lw=4, label= 'Susceptible')
	ax.plot(t, I, color = tab10[1], alpha=0.75, lw=4, label= 'Infected')
	ax.plot(t, R, color = tab10[2], alpha=0.75, lw=4, label= 'Recovered/Removed')

	ax.set_xlabel('Time (in days)')
	ax.set_ylabel('Confirmed Infected Cases')
	#ax.set_ylim(0,1.2)
	ax.yaxis.set_tick_params(length=0)
	ax.xaxis.set_tick_params(length=0)
	ax.grid(b=True, which='major', c='w', lw=2, ls='-')
	plt.legend(frameon=False, fontsize=12)
	for spine in ('top', 'right'):
	    ax.spines[spine].set_visible(False)

	return

def SIR_curves_scenarios(init_pop, beta_list, rnaught_list, x_range, 
	labels, g = 0.23, with_intervention = None, cumulative = False):
	#DEFINE PARAMETERS and INITIAL CONDITIONS here
	global beta, gamma, N, I0, R0, S0, t
	t = x_range
	gamma = g 		# gamma = mean recovery rate (in 1/days)
	N = init_pop        # Initial Susceptible population

	I0 = 4 #Initial Infected
	R0 = 0.0 #Initial Recovered
	S0 = N - I0 #Initial Susceptible

	y0 = S0, I0, R0

	plt.figure(figsize=(15,10));
	
	labels_rnaught = ['no intervention, R-naught: '] + ['with enhanced intervention, R-naught: ']*3

	if with_intervention == None:
		icounter = 0
		for b in beta_list:
			beta = b
			ret = odeint(sir_model, y0, t, args=(N, beta, gamma))
			S, I, R = ret.T
			if cumulative == True:
				plt.plot(t, np.cumsum(I), color = tab10[icounter], alpha=0.75, lw=4, \
					label = labels_rnaught[icounter] + str(rnaught_list[icounter]))
			else:
				plt.plot(t, I, color = tab10[icounter], alpha=0.75, lw=4, label= labels[icounter])
			icounter = icounter + 1

	else:
		intervene = with_intervention - 1
		icounter = 0
		icumulative = []
		for b in beta_list:
			if icounter == 0:
				beta = b
				ret = odeint(sir_model, y0, t, args=(N, beta, gamma))
				S, I, R = ret.T
				icumulative.append(np.cumsum(I))
				if cumulative == True:
					plt.plot(t, np.cumsum(I), color = tab10[icounter], alpha=0.75, lw=4, \
						label = labels_rnaught[icounter] + str(rnaught_list[icounter]))
				else:
					plt.plot(t, I, color = tab10[icounter], alpha=0.75, lw=4, \
						label= labels[icounter])

			elif icounter == 1:
				beta = b
				y0 = S[intervene], I[intervene], R[intervene]
				ret = odeint(sir_model, y0, t, args=(N, beta, gamma))
				S, I, R = ret.T
				I_adjust = icumulative[0][intervene-1] + np.cumsum(I)
				icumulative.append(I_adjust)
				if cumulative == True:
					plt.plot(t + intervene, I_adjust, color = tab10[icounter], alpha=0.75, lw=4, \
						label = labels_rnaught[icounter] + str(rnaught_list[icounter]))
				else:
					plt.plot(t + intervene, I, color = tab10[icounter], alpha=0.75, lw=4, label= labels[icounter])
			else:
				beta = b
				ret = odeint(sir_model, y0, t, args=(N, beta, gamma))
				S, I, R = ret.T
				I_adjust = icumulative[0][intervene-1] + np.cumsum(I)
				icumulative.append(I_adjust)
				if cumulative == True:
					plt.plot(t + intervene, I_adjust, color = tab10[icounter], alpha=0.75, lw=4, \
						label = labels_rnaught[icounter] + str(rnaught_list[icounter]))
				else:
					plt.plot(t + intervene, I, color = tab10[icounter], alpha=0.75, lw=4, label= labels[icounter])
			icounter = icounter + 1



	
	plt.xlabel('Time (in days)')
	plt.ylabel('Confirmed Infected Cases')
	#ax.set_ylim(0,1.2)
	plt.xlim(0, t[-1])
	plt.gca().yaxis.set_tick_params(length=0)
	plt.gca().xaxis.set_tick_params(length=0)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	plt.legend(frameon=False, fontsize=12)

	return