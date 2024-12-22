'''
groundwater model developement
'''
from typing import List, Dict, Tuple
import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
import os
  
def gw_model_core(
    x: np.ndarray,
    a : np.ndarray,
    b : np.ndarray,
    z : np.ndarray,
    c : np.ndarray,
    d: np.ndarray,
    z_tides : np.ndarray,
    h_init: np.ndarray,
    dt : Union[int, float],
    subterm_fname: Optional[str] = None,
) -> np.ndarray:

    #print("a", a, "b", b, "c", c, "d",d, "z", z, "z_tides", z_tides, "dt", dt)
    '''
    dh/dt = i-o

    1st tank:
    h1(t+1) = h1(t) + (R(t) - a1(h(t) - z1) - b1h1(t)) * dt

    2nd tank:
    h2(t+1) = h2(t) + (b1h1(t) - a2(h2(t) - z2) - b2h2(t)) * dt

    3rd tank (groundwater):
    h3(t+1) = h3(t) + (b2h2(t) - a3(h3(t) - z3) - b3h3(t) - c*AMP(t) + d(T(t) - z_tides)) * dt
    '''

    assert isinstance(x, np.ndarray)
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert isinstance(z, np.ndarray)
    assert isinstance(c, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z_tides, np.ndarray)
    assert isinstance(h_init, np.ndarray)
    assert isinstance(dt, (int, float))    #union
    assert h_init.shape == a.shape
    assert h_init.shape == b.shape
    assert c.shape == (1,)
    assert d.shape == (1,)
    assert h_init.shape == z.shape
    assert z_tides.shape == (1,)


    R = x[:, 0]
    Qwell = x[:, -2]
    Tides = x[:, -1]  
    tstep_number = R.shape[0]
    tank_number = h_init.shape[0]
    h_tank = np.zeros((tstep_number + 1, tank_number))
    h_tank[0, :] = h_init
    dt = 1
    
    Q_list = []
    for t in range(tstep_number):
        h_tank[t+1, :] = h_tank[t, :]

        # horizontal exchange Q = a(h - z)
        horizontal_exchange = a * (h_tank[t, :] - z)*  dt
        Q_list.append(horizontal_exchange)
        h_tank[t+1, :] -= horizontal_exchange

        # vertical flow P = bh
        vertical_flow = b * h_tank[t, :] * dt
        h_tank[t+1, :] -= vertical_flow

        # Rainfall
        h_tank[t+1, 0] += R[t] * dt

        # Pumping
        pumping = c * Qwell[t] * dt
        h_tank[t+1, :] -= pumping

        # Tides
        tides = d * (Tides[t] - z_tides) * dt
        h_tank[t+1, :] += tides

        # Recharge from previous tank
        h_tank[t+1, 1:] += vertical_flow[:-1]

    if subterm_fname is not None:
        # export the sub-terms of the model
        # h_tank, hgw, horizontal_exchange, verticle_exchange, pumping, tides, prev_discharge, R
        assert isinstance(subterm_fname, str)
        print('the subterm_frame', subterm_fname)

        # extract the first value of each array in the ndarray
        Q_list = np.asarray(Q_list)
        #print('Q_list', Q_list)
        Q1 = Q_list[:, 0]
        Q1 = Q1.tolist()
        Q2 = Q_list[:, 1]
        Q2 = Q2.tolist()
        Q3 = Q_list[:, -1]
        Q3 = Q3.tolist()

    return h_tank[1:, -1] # This return the last only which represents groundwater

   # x: np.ndarray,
def gw_model_shell(
    x: np.ndarray,
    *parameters,
    h_init = [],
    dt = Union[int, float],     
    **kwargs,
) -> np.ndarray:
    #print(parameters)
    assert isinstance (parameters, tuple)
    tank_number = 3
    #print(parameters)
    a = np.array(parameters[0:tank_number*1])
    b = np.array(parameters[tank_number * 1 : tank_number * 2])
    z = np.array(parameters[tank_number * 2 : tank_number * 3])
    c = np.array(parameters[tank_number * 3 : tank_number * 3 + 1])
    d = np.array(parameters[tank_number * 3 + 1 : tank_number * 3 + 2])
    z_tides = np.array(parameters[tank_number * 3 + 2 : tank_number * 3 + 3])
    dt = 1
    #print("h_init", h_init, "a", a, "b", b, "c", c, "d",d, "z", z, "z_tides", z_tides, "dt", dt)
    return gw_model_core(x, a, b, z, c, d,  z_tides, h_init, dt)


