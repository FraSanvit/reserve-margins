#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:06:29 2022

@author: fs
"""
import calliope
import create_input
import run

from dual_utils import duals_to_pickle, load_duals, process_system_balance_duals

import os
import ntpath

# SETTINGS

time_resolution ='1' # hours
model_year = '2018'
# mode = "plan_reserve_short"
mode = "plan-reserve-target-adder"

# SCENARIO DEFINITION

def get_scenario(base_scenario, projection_year, co2_scenario):
    if int(projection_year) == 2030:
        scenario = base_scenario + f",heat_techs_2030,renewable_techs_2030,transformation_techs_2030,coal_supply,fossil-fuel-supply"
    elif int(projection_year) == 2050:
        scenario = base_scenario  
        if co2_scenario == "current":
            scenario += f",fossil-fuel-supply"
    if co2_scenario=="compensation_abroad" or "no_compensation_abroad":
        scenario += f",ccs"
    return scenario

projection_year = '2050'
# link_cap_EU = "dynamic" # ["1x","5x", ..., "dynamic"]
# co2_scenario = "compensation_abroad" # ["neutral","current","neutral_extra","compensation_abroad","no_compensation_abroad"] # stick to this terminology since they are link to overrides

base_scenario = "reserve_margins, res_{time_resolution}h"

# base_scenario = "industry_fuel,extra-transport,heat,config_overrides,gas_storage,"\
#     f"link_cap_{link_cap_EU},freeze-hydro-capacities,res_{time_resolution}h,add-biofuel,"\
#     f"synfuel_transmission,{projection_year}_{co2_scenario},add-biogas,"\
#     f"{h2_station_cost}"

# scenario = get_scenario(base_scenario, projection_year, co2_scenario)
# extra_scenarios = ''
# scenarios_string = scenario + extra_scenarios

scenarios_string = ''


# MODEL RUN & SAVING RESULTS

path_to_model_yaml = f'model/model-{model_year}.yaml'
path_to_netcdf_of_model_inputs = f'model/{mode}-{time_resolution}h_inputs.nc'
path_to_netcdf_of_results=f'results/{mode}-{time_resolution}h.nc'

model_input = create_input.build_model(path_to_model_yaml, scenarios_string, path_to_netcdf_of_model_inputs)
model_run, duals, backend_model = run.run_model(path_to_netcdf_of_model_inputs, path_to_netcdf_of_results)


# DUALS POSTPROCESSING

system_balance_duals = duals['system_balance_constraint Constraint']
balance_duals = process_system_balance_duals(system_balance_duals)
balance_duals.set_index('timestep',inplace=True)
balance_duals.to_csv(f'duals/duals_{projection_year}-{mode}-{time_resolution}h.csv')

model_data = model_run._model_data
#model_run.to_csv(f"results/{mode}")

caps = model_run.get_formatted_array('energy_cap').to_pandas().T
caps[caps.index.isin(["ccgt","open_field_pv","wind_offshore","wind_onshore_monopoly"])].sum(axis=1).sum()
cap_gen = caps[caps.index.isin(["ccgt","open_field_pv","wind_offshore","wind_onshore_monopoly"])].sum().sum()

model_input = calliope.read_netcdf(path_to_netcdf_of_model_inputs)
backend_model._model_data["cap_value"].to_pandas()
backend_model._model_data.coords
backend_model._model_data.coords["loc_tech_cap_value_constraint"].values

backend_model.to_lp(f"model-{mode}.lp")
"""
if "group_target_reserve_share" in model_data: print("yes")

    reserve_margin_1:
        techs: [wind_offshore, wind_onshore_monopoly, wind_offshore, ccgt]
        locs: [N1, N2, N3, N4]
        target_reserve_share_min:
            electricity: 1.50  # (%)

"""


