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
mode = "operating_adder"

# SCENARIO DEFINITION

projection_year = '2050'
base_scenario = "reserve_margins, res_{time_resolution}h"
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


# MODEL OUTPUT

model_data = model_run._model_data
model_run.to_csv(f"results/{mode}")

caps = model_run.get_formatted_array('energy_cap').to_pandas().T
caps[caps.index.isin(["ccgt","open_field_pv","wind_offshore","wind_onshore_monopoly"])].sum(axis=1).sum()
cap_gen = caps[caps.index.isin(["ccgt","open_field_pv","wind_offshore","wind_onshore_monopoly"])].sum().sum()


# MODEL INPUT

model_input = calliope.read_netcdf(path_to_netcdf_of_model_inputs)


# LP FILE

backend_model.to_lp(f"lp-files/model-{mode}.lp")


backend_model._model_data["cap_value"].to_pandas()
backend_model._model_data.coords
backend_model._model_data.coords["loc_tech_cap_value_constraint"].values