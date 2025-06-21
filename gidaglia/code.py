import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numba
from numba import njit


########################################################### Reading the data ##########################################################


file_path = 'Copie de Suivi batch 2022 (copy).xlsx'
df = pd.read_excel(file_path)

starting_date = df['Date début']
starting_date = starting_date[658:759]

starting_time = df['Heure début']
starting_time = starting_time[658:759]

ending_time = df['Heure fin']
ending_time = ending_time[658:759]

viscosity = df['Viscosité(mPa.s)']
viscosity = viscosity[658:759]




starting_time_fixed = starting_time.astype(str).str.strip().apply(lambda x: x if len(x.split(':')) == 3 else x + ':00')
ending_time_fixed = ending_time.astype(str).str.strip().apply(lambda x: x if len(x.split(':')) == 3 else x + ':00')


start_datetime = pd.to_datetime(starting_date.astype(str) + ' ' + starting_time_fixed)
end_datetime = pd.to_datetime(starting_date.astype(str) + ' ' + ending_time_fixed)


end_datetime = end_datetime.where(end_datetime >= start_datetime, end_datetime + pd.Timedelta(days=1))

timeline = []
viscosity_values = []

for i in range(len(start_datetime)):
   
    timeline.append(start_datetime.iloc[i])
    viscosity_values.append(viscosity.iloc[i])

    timeline.append(end_datetime.iloc[i])
    viscosity_values.append(viscosity.iloc[i])

    if i < len(start_datetime) - 1:
        next_start = start_datetime.iloc[i+1]
        if end_datetime.iloc[i] < next_start:
            timeline.append(end_datetime.iloc[i])
            viscosity_values.append(1.0)  

            timeline.append(next_start)
            viscosity_values.append(1.0)


df_plot = pd.DataFrame({
    'datetime': timeline,
    'viscosity': viscosity_values
}).sort_values('datetime')

df_plot = df_plot.sort_values('datetime').reset_index(drop=True)

# plt.figure(figsize=(14,6))
# plt.step(df_plot['datetime'][0:100].to_numpy(), df_plot['viscosity'][0:100].to_numpy(), where='post')
# plt.xlabel('Time')
# plt.ylabel('Viscosity (mPa.s)')
# plt.title('Viscosity over Time')
# plt.grid()
# plt.show()

df_plot['datetime'] = pd.to_datetime(df_plot['datetime'])

df_plot = df_plot.sort_values('datetime').reset_index(drop=True)

start_time = pd.Timestamp('2022-08-01 00:00:00')
end_time = df_plot['datetime'].max()

time_seconds = np.arange(0, int((end_time - start_time).total_seconds()) + 180, 180)

time_stamps = start_time + pd.to_timedelta(time_seconds, unit='s')

viscosity_series = df_plot.drop_duplicates('datetime').set_index('datetime')['viscosity']

viscosity_filled = viscosity_series.reindex(time_stamps, method='ffill')

time_data = time_seconds
viscosity_data = viscosity_filled.values

file_path = 'Copie de Tags Main Pipeline.xlsx'  


df = pd.read_excel(file_path, header=[0,1])


df.columns = [' '.join(col).strip() for col in df.columns.values]


columns_to_extract = [
    'Station Terminal Pression Station terminal',
    'Station Terminal Débit Station terminal',
    'Densité Station tete Densité sortie train A'
]

# Extract
extracted_df = df[columns_to_extract]

# Save the result
extracted_df.to_excel('extracted_columns.xlsx', index=False)


pression_array1 = extracted_df['Station Terminal Pression Station terminal'].to_numpy()
debit_array1 = extracted_df['Station Terminal Débit Station terminal'].to_numpy()
densite_array1 = extracted_df['Densité Station tete Densité sortie train A'].to_numpy()
pression_array1 = pression_array1[1:]
debit_array1 = debit_array1[1:]
densite_array1 = densite_array1[1:]
time_data2 = np.arange(len(densite_array1))*180


pression_array = pression_array1[7500:10000]
debit_array = debit_array1[7500:10000]
densite_array = densite_array1[7500:10000]
time_data2 = time_data2[7500:10000]


columns_to_extract2 = [
    'Pression Station tete  pression sortie TRAIN A',
    'Station Terminal Densité Station terminal'
]

# Extract
extracted_df2 = df[columns_to_extract2]

# Save the result
extracted_df2.to_excel('extracted_columns2.xlsx', index=False)

pression_to_compare1 = extracted_df2['Pression Station tete  pression sortie TRAIN A'].to_numpy()
density_to_comare1 = extracted_df2['Station Terminal Densité Station terminal'].to_numpy()


pression_to_compare1 = pression_to_compare1[1:]
density_to_compare1 = density_to_comare1[1:]

pression_to_compare = pression_to_compare1[7500:10000]
density_to_compare = density_to_compare1[7500:10000]

idx_start = np.abs(time_data - time_data2[0]).argmin()
idx_end = np.abs(time_data - time_data2[-1]).argmin()

time_data2 = time_data2 -time_data2[0]

testtt = debit_array.copy()
viscosity_array_prime = viscosity_data[idx_start:idx_end+1]

viscosity_array = viscosity_array_prime.copy()

for i in range(len(viscosity_array)):
    if densite_array[i]>1400 and viscosity_array[i]<5:
        viscosity_array[i] = 16
    if densite_array[i]<1400 and viscosity_array[i]>5:
        viscosity_array[i] = 1

for i in range(10,len(debit_array)-10):
    if debit_array[i]>5000:
        debit_array[i] = debit_array[i+1]
    if np.abs(debit_array[i]-debit_array[i-1])>280 and debit_array[i-10:i+10].all()>3600 and( i<870 or i>1010):
        debit_array[i] = debit_array[i-1]




fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# Plot 1: Rate Flow
axs[0].plot(time_data2/3600, testtt, label="Rate Flow ", color='b')
axs[0].plot(time_data2/3600, debit_array, label="Rate Flow Without Outliers", color='r')
axs[0].set_ylabel("Rate Flow")
axs[0].set_title("Rate Flow vs. Time")
axs[0].legend()
axs[0].grid(True)

# Plot 2: Density
axs[1].plot(time_data2/3600, densite_array, label="RHO (density)", color='b')
axs[1].set_ylabel("Density")
axs[1].set_title("Density vs. Time")
axs[1].legend()
axs[1].grid(True)

# Plot 3: Viscosity
axs[2].plot(time_data2/3600, viscosity_array, label="Viscosity", color='b')
axs[2].set_ylabel("Viscosity")
axs[2].set_title("Viscosity vs. Time")
axs[2].legend()
axs[2].grid(True)

# Plot 4: Pressure
axs[3].plot(time_data2/3600, pression_array, label="P (Pressure)", color='b')
axs[3].set_xlabel("Time (h)")
axs[3].set_ylabel("Pressure (kPas)")
axs[3].set_title("Pressure vs. Time")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.savefig('data.png')

mesure_time = time_data2
rho_in_mes = densite_array

mu_in_mes = viscosity_array*1e-3
Q_out_mes = debit_array/3600
P_out_mes = 1000 * pression_array


file_path = "profil pipe principal.xlsx"

df = pd.read_excel(file_path, sheet_name="Feuil1", header=None)

df.columns = ["Ignore", "L", "E"]

df = df.drop(columns=["Ignore"])

df = df.dropna()

L_array = df["L"].values
E_array = df["E"].values

L_array = 1000 * L_array

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% The data is Ready %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

##################################################################################################################################

########################################################### The Solver ###########################################################
L = 187124  
D = 0.855   
A = np.pi * (D**2) / 4  
N = 500    
dx = L / N  
eps = .0213e-3

x = np.linspace(0, L, N)  

Elevation = np.interp(x, L_array, E_array)



theta = np.arctan(np.diff(Elevation) / np.diff(x))
theta = np.append(theta, theta[-1])


U_in = Q_out_mes[0]/(A)
# print(U_in)


Q_in = U_in * A  
rho_s =  1400
rho_w = 1000

mu_w = 1e-3  
mu_s = 0.0102  


rho_in = 1400 * np.ones(N)    
mu_in = 16e-3 * np.ones(N)

a = 1.25
Tf = mesure_time[0]+120*3600
dt = 2
g = 9.81  
lam = 0.1



def m_vector(rho,A,dx):
    m_vec = A*dx*rho
    return m_vec

@njit
def I_vector(v,rho,A,dx):
    I_vec = np.zeros_like(rho)
    for i in range(len(I_vec)):
        I_vec[i] = v*A*dx*rho[i]
    return I_vec

m_vec_0 = m_vector(rho_in, A, dx)
I_vec_0 = I_vector(U_in, rho_in, A, dx)

@njit
def Reynold(rho, u, D, mu):
    Re = np.zeros(N)
    for i in range(N):
        Re[i] = rho[i]*u*D/mu[i]
    return Re

@njit
def compute_P(P, A , g,theta, U_old, U_new, U, m_i, pressure_drop, dt,P_out):
    P_new = np.zeros_like(P)
    P_new[-1] = P_out
    for i in range(N-2, -1, -1):
        P_new[i] = P_new[i+1] + m_i[i]/A * (np.abs(g)*np.sin(theta[i]) + (U_new-U)/dt) + pressure_drop[i]
        
    if U < 1 or U_new <1 or U_old <1:
        P_new[0] = 0
    return P_new



#@njit
def Friction(a,Re,D,eps):
    den = np.zeros(N)
    fr = np.zeros(N)
    for i in range(N):
        den[i] = np.log(eps/(3.7*D)+5.74/(Re[i]**0.9)) ** 2
        fr[i] = a/den[i]
                              
    return fr

@njit
def sigmoid(fa,fb,ta,tb,lam,t):
    return fa + (fb-fa) / (1 + np.exp((0.5 * (ta+tb)-t)/(lam*dt)))



Re_s = Reynold(rho_in, U_in, D, mu_in)
f_D = Friction(a, Re_s, D, eps)
 
pressure_drop = dx * f_D * rho_in * U_in**2 / (2 * D)
@njit
def compute_dt(U, mu, dx, Cfl):
    if U<1e-3:
        dt_adv = 10
    else:
        dt_adv = Cfl * dx / abs(U)
    dt_diff = Cfl * dx**2 / np.max(mu)
    return min(200, dt_adv)#, dt_diff)

P_0 = np.zeros(N)  
U = U_in  

P_0[-1] = P_out_mes[0]



def solver():

    dt = 10
    
    m_vec = m_vec_0.copy()
    I_vec = I_vec_0.copy()
    mu = mu_in.copy()
    P = P_0.copy()
    rho = rho_in.copy()
    #q = q.copy()
    U = U_in
    Q = Q_in

    PHI = 0
    
    
    m_vec_new, I_vec_new, mu_new, rho_new, P_new = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    U_new = U
    Q_new = Q
    U_old = U
    time = mesure_time[0]
    U_data , rho_inlet_data, rho_outlet_data, mu_inlet_data, mu_outlet_data , Q_inlet_data, Q_outlet_data, P_inlet_data, P_outlet_data = [],[],[],[],[],[],[],[],[]
    plot_time = []
    j=0
    number_of_hours = 0
    previous_hour = -1
    while time < Tf:
        
        idx = np.argmin(np.abs(mesure_time - time))
        idx2 = np.argmin(np.abs(mesure_time - (time+dt)))
        

        rho_b = rho_in_mes[idx]
        mu_b = mu_in_mes[idx]
        P_out = P_out_mes[idx]
        
        m_vec_new[0] = A*dx*rho_b
        m_vec_new[1:N] = m_vec[1:N] - (dt/dx) * U * (m_vec[1:N] - m_vec[0:N-1])
        
        
        I_vec_new[:] = U*m_vec[:]  
        
        Re_s = Reynold(rho, U, D, mu)
        f_D = Friction(a, Re_s, D, eps)

        
        pressure_drop = dx * f_D * rho * (U)**2 / (2 * D) + 0.0033 * PHI * 1/2 * rho *U**2
        
        
        U_new = Q_out_mes[idx2]/(A) 
        

        P_new = compute_P(P, A, g, theta,U_old, U_new, U, m_vec_new, pressure_drop, dt,P_out)
        
        
        mu_new[0] = mu_b
        mu_new[1:N] = mu[1:N] - dt/dx * U *(mu[1:N]-mu[0:N-1])

        rho_new = m_vec_new / (A*dx)
        
        for i in range(N):
            rho_new[i] = sigmoid(rho_new[i], rho[i], time, time+dt, lam, time)
            mu_new[i] = sigmoid(mu_new[i], mu[i], time, time+dt, lam, time)


        current_hour = int(time // 3600) + 1
        if current_hour > previous_hour:
            print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% {current_hour}h simulated %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            previous_hour = current_hour
            #print((max(pressure_drop)))
            # Plot rho_data
            

        dt = compute_dt(U, mu, dx, Cfl=.99)
        time += dt
        j += 1
        m_vec, I_vec, mu, rho, P = m_vec_new.copy(), I_vec_new.copy(), mu_new.copy(), rho_new.copy(), P_new.copy()
        U_old = U
        U = U_new
        Q_new = U_new * A
        Q = Q_new
        U_data.append(U)
        rho_inlet_data.append(rho[0])
        rho_outlet_data.append(rho[-1])
        mu_inlet_data.append(mu[0]*1000)
        mu_outlet_data.append(mu[-1]*1000)
        P_inlet_data.append(P[0]/1000)
        P_outlet_data.append(P[-1]/1000)
        Q_inlet_data.append(Q_new*3600)
        Q_outlet_data.append(Q_new*3600)
        plot_time.append(time)
    
    return plot_time, U, mu, rho, P, U_data , rho_inlet_data, rho_outlet_data, mu_inlet_data, mu_outlet_data , Q_inlet_data, Q_outlet_data, P_inlet_data, P_outlet_data

plot_time, U_final, mu_final, rho_final, P_final, U_data , rho_inlet_data, rho_outlet_data, mu_inlet_data, mu_outlet_data , Q_inlet_data, Q_outlet_data, P_inlet_data, P_outlet_data = solver()
for i in range(1,len(P_inlet_data)):
    if P_inlet_data[i] <-5:
        P_inlet_data[i] = 0
    if P_inlet_data[i] > 5500:
        P_inlet_data[i] = P_inlet_data[i-1]


##################################################################################################################################

########################################################### The Results ##########################################################


interp_func = interp1d(mesure_time, pression_to_compare, kind='linear', fill_value='extrapolate')

# Interpolate onto the new time vector
pression_interpolated = interp_func(np.array(plot_time))

interp_func = interp1d(mesure_time, density_to_compare, kind='linear', fill_value='extrapolate')

# Interpolate onto the new time vector
density_interpolated = interp_func(np.array(plot_time))

# Create a time array


fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# Plot Pressure data

axs[0].plot(np.array(plot_time)/3600, pression_interpolated, label="P (Pressure), Measured", color='purple')
axs[0].plot(np.array(plot_time)/3600, np.array(P_inlet_data), '--', label="P (Pressure), Present Work", color='black')
axs[0].set_xlabel("Time (h)")
axs[0].set_ylabel("Pressure (kPa)")
axs[0].set_title("Pressure vs. Time")
axs[0].legend()
axs[0].grid(True)

# Plot Density data
axs[1].plot(np.array(plot_time)/3600, density_interpolated, label="Density (rho), Measured", color='purple')
axs[1].plot(np.array(plot_time)/3600, np.array(rho_outlet_data), '--', label="Density (rho), Present Work", color='black')
axs[1].set_xlabel("Time (h)")
axs[1].set_ylabel("Density (kg/m³)")
axs[1].set_title("Density vs. Time")
axs[1].legend()
axs[1].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.savefig('Results')


