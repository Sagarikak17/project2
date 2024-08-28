import math
import numpy as np

def read_data_from_file(input):
    data = []
    with open(input,'r') as file:
        for line in file:
            temp_celsius,pressure,molality = map(float, line.strip().split())
            data.append([temp_celsius,pressure,molality])
    return data


def read_constants_from_file(constants):
    with open(constants, 'r') as file:
        constants = file.readlines()
    Pc,Tc,a1, a2,a3,a4, a5, a6,R,PcCO2,TcCO2, wCO2,delta1,delta2,n,Mwh2o,tau,beta = map(float, constants)
    return Pc, Tc, a1,a2, a3,a4, a5,a6, R, PcCO2, TcCO2, wCO2,delta1,delta2,n,Mwh2o,tau,beta

def calculate_A1(temp_celsius):
    A1 = (3.2891-2.391e-3*temp_celsius +2.8446e-4*temp_celsius**2-2.82e-6*temp_celsius**3+8.477e-9*temp_celsius**4)
    return A1

def calculate_A2(temp_celsius):
    A2 = 6.245e-5-3.913e-6*temp_celsius-3.499e-8*temp_celsius**2+7.942e-10 *temp_celsius**3-3.299e-12*temp_celsius**4
    return A2

def calculate_Bh2o(temp_celsius):
    Bh2o = 19654.32+147.037*temp_celsius-2.2155*temp_celsius**2+1.0478e-2*temp_celsius**3-2.2789e-5*temp_celsius**4
    return Bh2o


def calculate_V0(temp_celsius):
    numerator = 1+18.1597*10**-3*temp_celsius
    denominator = 0.9998+ (18.2249 * 10**-3)*temp_celsius-(7.9222*10**-6)*temp_celsius**2-(55.4485*10**-9)*temp_celsius**3+(149.7562*10**-12)*temp_celsius**4-(393.2952*10**-15)*temp_celsius**5
    V0 = numerator / denominator
    return V0

def calculate_rho(temp_celsius, pressure):
    A1 = calculate_A1(temp_celsius)
    A2 = calculate_A2(temp_celsius)
    Bh2o = calculate_Bh2o(temp_celsius)
    V0 = calculate_V0(temp_celsius)
    rho = 1/(V0- V0*pressure/(Bh2o+ A1*pressure+ A2*pressure**2))
    return rho

def calculate_temp_kelvin(temp_celsius):
    temp_kelvin = temp_celsius + 273.15
    return temp_kelvin

def calculate_Ps_Pc(temp_kelvin,Tc,a1,a2,a3,a4,a5,a6):
    x = 1-temp_kelvin/Tc
    ln_ratio = (a1*x + a2*x**1.5+ a3*x**3 + a4*x**3.5 + a5*x**4 + a6*x**7.5)
    ln_Ps_Pc = ln_ratio*(Tc/temp_kelvin)
    Ps = math.exp(ln_Ps_Pc) * Pc
    return Ps

def calculate_fugacity_H20(pressure,Ps, rho, temp_kelvin, R):
    fh2o = Ps * math.exp(18.0152*(pressure - Ps)/(rho * R * temp_kelvin))
    return fh2o

def calculate_mCO2(wCO2):
    mCO2 = 0.37464+ 1.54226*wCO2 - 0.26992 * wCO2**2
    return mCO2

def calculate_a(R,TcCO2,PcCO2,mCO2,temp_kelvin):
    a = (0.457236 *R**2* TcCO2**2 / PcCO2) *(1 +mCO2 *(1 - math.sqrt(temp_kelvin / TcCO2)))**2
    return a

def calculate_b(R,TcCO2,PcCO2):
    b = 0.077796 *R* TcCO2/PcCO2
    return b

def calculate_A(a, pressure, R ,temp_kelvin):
    A = a*pressure/(R*temp_kelvin)**2
    return A
    
def calculate_B(b, pressure, R, temp_kelvin):
    B = b*pressure/(R*temp_kelvin)
    return B

def solve_cubic(A,B):
    coeffs = [1,-1+B,A-2*B-3*B**2,-A*B+B**2 +B**3]
    roots = np.roots(coeffs)
    zl = min(roots)
    zg = max(roots)
    return zg,zl

def select_root(A, B, zg, zl, delta1, delta2):
    if zg-zl + np.log(zl-B/(zg - B))-A/(B*(delta2 - delta1))*np.log((zl + delta1*B)*(zg+ delta2*B)/(zl+delta2*B)*(zg +delta1*B)) > 0:
        return zl
    else:
        return zg
    
def calculate_z(zl,zg):
    Z = select_root(A,B,zg, zl,delta1, delta2)
    return Z


def calculate_phi(A,B, Z, delta1,delta2):
    phi = np.exp(Z-1-np.log(Z-B)-(A/(B*(delta2-delta1)))*np.log((Z+delta2*B)/(Z + delta1*B)))
    return phi



def calculate_delB(tau,beta,temp_kelvin):
    delB = tau + beta*(1000/temp_kelvin)**0.5
    return delB


def calculate_hi(n,fh2o,R,temp_kelvin,Mwh2o,tau,beta,rho,delB):
    hi = np.exp((1-n)*np.log(fh2o) + n*np.log(R*temp_kelvin*rho/Mwh2o) + 2*rho*delB)
    return hi

def calculate_lamda(temp_kelvin,pressure):
    lamda = -0.0652869 + 1.6790636e-4*temp_kelvin + 40.838951/temp_kelvin -3.9266518e-2*pressure/temp_kelvin + 2.1157167e-2*pressure/(630-temp_kelvin)+6.5486487e-6*temp_kelvin*np.log(pressure)
    return lamda

def calculate_epsilon(temp_kelvin,pressure):
    epsilon = -1.144624e-2 + 2.8274958e-5*temp_kelvin + 1.3980876e-2*pressure/temp_kelvin - 1.4349005e-2*pressure/(630-temp_kelvin)
    return epsilon

def calculate_gamma(lamda,epsilon,molality):
    gamma = np.exp(2*molality*lamda + 2*molality**2*epsilon)
    return gamma

def calculate_kco2(hi,gamma,pressure,phi):
    kco2 = hi*gamma/(pressure*phi)
    return kco2

def calculate_k0h2o(temp_celsius):
    k0h2o = 10**(-2.209 + 3.097e-2*temp_celsius - 1.098e-4*temp_celsius**2 + 2.048e-7*temp_celsius**3)
    return k0h2o

def calculate_kh2o(k0h2o,fh2o,pressure,R,temp_kelvin):
    kh2o = k0h2o*np.exp((pressure - 1)*18.18/(R * temp_kelvin))/(fh2o*pressure)
    return kh2o

def calculate_yh2o(kco2,kh2o):
    num1 = 1-1/kco2
    den1 = 1/kh2o-1/kco2
    yh2o = num1/den1
    return yh2o

def calculate_yco2(yh2o):
    yco2 = 1/(1 + yh2o)
    return yco2

def calculate_xco2(yco2,kco2):
    xco2 = yco2/kco2
    return xco2

def calculate_xco2percent(xco2):
    xco2percent = 100*xco2
    return xco2percent

input_file = "input.txt"
constants_file = "constants.txt"
data = read_data_from_file(input_file)
Pc, Tc, a1, a2, a3, a4, a5, a6, R, PcCO2, TcCO2, wCO2,delta1,delta2,n,Mwh2o,tau,beta = read_constants_from_file(constants_file)

for entry in data:
    temp_celsius, pressure, molality= entry
    A1 = calculate_A1(temp_celsius)
    A2 = calculate_A2(temp_celsius)
    Bh2o = calculate_Bh2o(temp_celsius)
    V0 = calculate_V0(temp_celsius)
    rho = calculate_rho(temp_celsius, pressure)
    temp_kelvin = calculate_temp_kelvin(temp_celsius)
    Ps = calculate_Ps_Pc(temp_kelvin,Tc,a1,a2,a3,a4,a5,a6)
    fh2o = calculate_fugacity_H20(pressure,Ps, rho, temp_kelvin, R)
    mCO2 = calculate_mCO2(wCO2)
    a = calculate_a(R,TcCO2,PcCO2,mCO2,temp_kelvin)
    b = calculate_b(R,TcCO2,PcCO2)
    A = calculate_A(a, pressure, R ,temp_kelvin)
    B = calculate_B(b, pressure, R, temp_kelvin)
    roots = solve_cubic(A,B)
    zl = min(roots)
    zg = max(roots)
    Z = select_root(A, B, zg, zl, delta1, delta2)
    phi = calculate_phi(A, B, Z, delta1, delta2)
    delB = calculate_delB(tau,beta,temp_kelvin)
    hi = calculate_hi(n,fh2o,R,temp_kelvin,Mwh2o,tau,beta,rho,delB)
    lamda = calculate_lamda(temp_kelvin,pressure)
    epsilon = calculate_epsilon(temp_kelvin,pressure)
    gamma = calculate_gamma(lamda,epsilon,molality)
    kco2 = calculate_kco2(hi,gamma,pressure,phi)
    k0h2o = calculate_k0h2o(temp_celsius)
    kh2o = calculate_kh2o(k0h2o,fh2o,pressure,R,temp_kelvin)
    yh2o = calculate_yh2o(kco2,kh2o)
    yco2 = calculate_yco2(yh2o)
    xco2 = calculate_xco2(yco2,kco2)
    xco2percent = calculate_xco2percent(xco2)
    print(f"FOR TEMPERATURE {temp_celsius}C & FOR PRESSURE {pressure}bar:" )
    '''print( "Value of V0:", V0)
    print("Value of Bh2o:", Bh2o)
    print("Value of A1:", A1)
    print("Value of A2:", A2)'''
    print("Value of rho:", rho)
    print("Value of Ps:", Ps)
    print("Value of fh2o:", fh2o)
    '''print("mco2:", mCO2)
    print("value of a:", a)
    print("value of b:", b)
    print("value of A:", A)
    print("value of B:", B)
    print("zl:", zl)
    print("zg:", zg)'''
    print("compressibility factor Z:",Z)
    print("fugacity coeff of co2(phi):", phi.real)
    print("delB", delB)
    print("henry constant for dissolved co2(hi):", hi)
    '''print("lamda", lamda)
    print("epsilon",epsilon)'''
    print("activity coeff of co2(gamma):", gamma)
    print("kco2",kco2.real)
    '''print("k0h2o", k0h2o)'''
    print("kh2o",kh2o)
    print("yh2o", yh2o.real)
    print("xco2", xco2.real)
    print("xco2(%)", xco2percent.real)
