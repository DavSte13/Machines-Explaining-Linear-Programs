import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer


class PlexPlain(nn.Module):
    """
    # TOTEX: total expenditures (CAPEX + OPEX)
    # CAPEX: capital expenditures (investment in e.g. machines, buildings ...)
    # OPEX: operational expenditures (running expenditures for a working business operation)

    Torch module for the cvxpylayers emulation of the PlexPlain model.
    The model consists of four different parameters:
        cost of the photovoltaic:       cost_pv
        cost of battery storage:        cost_bat
        price of power from the grid:   cost_buy
        energy demand:                  dem_tot

        There is also a lifetime (years) given.
        The cost of the photovoltaic and the battery storage are then divided by the lifetime

    the preset information (can also be considered as parameters):
        availability_pv: percentage of photovoltaic effectiveness per hour
        demand_val: percentage based split of the energy demand to hours

    variables:
        EnergyPV
        Demand
        EnergyBattery
        EnergyBattery_IN
        EnergyBattery_OUT
        EnergyBuy
        CAP_PV
        CAP_BAT

    input settings:
        CostBuy = pyo.Var(within=pyo.Reals)
        CostPV = pyo.Var(within=pyo.Reals)
        CostBat = pyo.Var(within=pyo.Reals)

    parameters: lower case, variables: upper case

    objective: cost_pv * CAP_PV + cost_buy * sum(ENERGY_BUY[i] for i in time) + cost_bat * CAP_BAT (minimize)

    constraints:

    # PV upper limit (i-times)
    EnergyPV[i] <= CAP_PV * availability_pv[i]          (availability_pv[i] is given and fixed)

    # battery upper limit (i-times)
    EnergyBattery[i] <= CAP_BAT

    # Battery level t=0 == t=T
    EnergyBattery[0] = EnergyBattery[-1] - EnergyBattery_OUT[0] + EnergyBattery_IN[0]

    # power demand (i-times)
    Demand[i] == dem_tot * DemandVal[i]                 (DemandVal[i] is given and fixed -> percentage based split of  the total demand into single hours)

    # battery equation (i-1-times)
    EnergyBattery[i] == EnergyBattery[i-1] - EnergyBattery_OUT[i] + EnergyBattery_IN[i]

    # energy equation (i-times)
    Demand[i] == EnergyBuy[i] + EnergyBattery_OUT[i] - EnergyBattery_IN[i] + EnergyPV[i]

    :param reduce_dimension: Used to generate a reduced model for the occlusion attributions.
    :param month: The current month to consider (only for evaluations with occlusion).
    :param use_days: Reduce the problem size for the gradient based methods to speedup computation.
    """

    def __init__(self, reduce_dimension=False, month=1, use_days=False):
        super().__init__()
        if reduce_dimension:
            h_per_year = 1095 - 91  # 3 time steps per day, but with one month less
        elif use_days:
            h_per_year = 1095  # 3 time steps per day
        else:
            h_per_year = 8760  # full time steps

        availability_pv = np.genfromtxt('evaluation/plexplain_data/PVavail.csv', delimiter='\n')
        demand_values = np.genfromtxt('evaluation/plexplain_data/demand.csv', delimiter='\n')

        if reduce_dimension:
            availability_pv = np.add.reduceat(availability_pv, np.arange(0, 8760, 8))
            demand_values = np.add.reduceat(demand_values, np.arange(0, 8760, 8))

            mask = [i < (month - 1) * 91 or i > month * 91 - 1 for i in range(1095)]
            availability_pv = availability_pv[mask]
            demand_values = demand_values[mask]
        elif use_days:
            availability_pv = np.add.reduceat(availability_pv, np.arange(0, 8760, 8))
            demand_values = np.add.reduceat(demand_values, np.arange(0, 8760, 8))

        # create the CVXPY problem
        # create the variables and parameters
        inputs = cp.Parameter(4)  # cost_pv, cost_bat, cost_buy, dem_tot

        energy_pv = cp.Variable(shape=h_per_year, nonneg=True)
        energy_battery = cp.Variable(shape=h_per_year, nonneg=True)
        energy_battery_in = cp.Variable(shape=h_per_year, nonneg=True)
        energy_battery_out = cp.Variable(shape=h_per_year, nonneg=True)
        energy_buy = cp.Variable(shape=h_per_year, nonneg=True)
        cap_pv = cp.Variable(shape=1, nonneg=True)
        cap_bat = cp.Variable(shape=1, nonneg=True)

        # matrix for the energy constraint
        shift_matrix = np.eye(h_per_year, k=-1)[1:]
        constraints = [
            energy_battery[0] == energy_battery[-1] - energy_battery_out[0] + energy_battery_in[0],
            energy_battery[1:] == (shift_matrix @ energy_battery).T - energy_battery_out[1:] + energy_battery_in[1:],
            inputs[3] * demand_values == energy_buy + energy_battery_out - energy_battery_in + energy_pv,
            energy_pv <= cap_pv[0] * availability_pv,
            energy_battery <= np.ones(h_per_year, ) * cap_bat[0],
        ]

        obj = cp.Minimize(inputs[0] * cap_pv[0] + inputs[2] * cp.sum(energy_buy) + inputs[1] * cap_bat[0])

        prob = cp.Problem(obj, constraints)
        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[inputs],
                                     variables=[energy_pv, energy_battery, energy_battery_in,
                                                energy_battery_out, energy_buy, cap_bat, cap_pv])

    def forward(self, inputs, use_sol=False):
        """
            Forward the inputs to the cvxpylayer.
            If use_sol equals True, the optimal solution is returned. Otherwise, the TOTEX value (the objective
            function value) is computed.
        """
        solution = self.cvxpylayer.forward(inputs)
        if use_sol:
            return solution
        else:
            result = inputs[0] * solution[6] + inputs[2] * torch.sum(solution[4]) + inputs[1] * solution[5]
            return result
