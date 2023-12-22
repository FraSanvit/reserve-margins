import argparse

import pyomo.core as po
import pyomo.environ as pyo
import numpy as np
import calliope
import pandas as pd

import logging

from calliope.backend.pyomo.util import (
    get_param,
    split_comma_list,
    get_timestep_weight,
    invalid,
)


def run_model(path_to_model, path_to_output):

    calliope.set_log_verbosity("info", include_solver_output=True, capture_warnings=True)
    model = calliope.read_netcdf(path_to_model)

    model = update_constraint_sets(model)

    model.run(build_only=True)
    
    # model._backend_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    add_eurocalliope_constraints(model)
    model._backend_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    new_model = model.backend.rerun()
    
    duals = {} 
    for c in model._backend_model.component_objects(pyo.Constraint, active=True):
        duals[("{} Constraint".format(c))] = []
        for index in c:
            duals["{} Constraint".format(c)].append(("{}".format(index), model._backend_model.dual[c[index]]))
        duals["{} Constraint".format(c)] = pd.DataFrame(duals["{} Constraint".format(c)])

    if new_model.results.attrs.get('termination_condition', None) not in ['optimal', 'feasible']:
        calliope.exceptions.BackendError("Problem is non optimal, not saving anything.")

    new_model.to_netcdf(path_to_output)
    
    return (new_model,duals,model)

"""
def run_model(path_to_model, path_to_output):

    calliope.set_log_verbosity("info", include_solver_output=True, capture_warnings=True)
    model = calliope.read_netcdf(path_to_model)

    model = update_constraint_sets(model)

    model.run(build_only=True)
    
    model._backend_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    add_eurocalliope_constraints(model)
    
    model.to_netcdf("prova.nc")
    # return (model)
"""
def update_constraint_sets(model):

    if "energy_cap_max_time_varying" in model._model_data.data_vars:
        print("Adding production_max_time_varying constraint set")
        model._model_data.coords["loc_tech_carrier_production_max_time_varying_constraint"] = [
            loc_tech for loc_tech in model._model_data.loc_techs.values
            if model.inputs.energy_cap_max_time_varying.loc[{"loc_techs": loc_tech}].notnull().all()
        ]
        print(
            f"{len( model._model_data.loc_tech_carrier_production_max_time_varying_constraint)}"
            " items in set"
        )
        for _set in [
            "loc_tech_carriers_carrier_production_max_constraint",
            "loc_techs_carrier_production_max_conversion_plus_constraint"
        ]:
            print(f"Removing production_max_time_varying loc::techs from {_set}")
            model._model_data.coords[_set] = [
                loc_tech_carrier for loc_tech_carrier
                in model._model_data[_set].values
                if loc_tech_carrier.rsplit("::", 1)[0] not in
                model._model_data.loc_tech_carrier_production_max_time_varying_constraint
            ]
           
    if "passenger_car_ev_battery" in model._model_data.techs:
        print("Adding storage_ev_constraint set")
        model._model_data.coords["loc_tech_ev_storage_constraint"] = [
            loc_techs_store
            for loc_techs_store in model._model_data.loc_techs_store.values
            if (model.inputs.storage_cap_max.loc[{"loc_techs_store": loc_techs_store}].notnull()) and
            (loc_techs_store.split("::")[1]=="passenger_car_ev_battery") 
        ]

        print(
            f"{len(model._model_data.loc_tech_ev_storage_constraint)}"
            " items in set"
        )
      
    if "cb" in model._model_data.data_vars:
        print("Adding chp_cb constraint set")

        model._model_data.coords["loc_techs_chp_extraction_cb_constraint"] = [
            loc_tech
            for loc_tech in model._model_data.loc_techs_conversion_plus.values
            if model._model_data.cb.loc[{"loc_techs": loc_tech}].notnull()
            and (
                "energy_cap_ratio" not in model._model_data.data_vars or
                model._model_data.energy_cap_ratio.loc[{"loc_techs": loc_tech}].isnull()
            )
        ]
        print(
            f"{len(model._model_data.loc_techs_chp_extraction_cb_constraint)}"
            " items in set"
        )
        print(
            "Removing conversion_plus loc::techs from "
            "loc_techs_balance_conversion_plus_out_2_constraint"
        )
        model._model_data.coords["loc_techs_balance_conversion_plus_out_2_constraint"] = [
            loc_tech for loc_tech
            in model._model_data.loc_techs_balance_conversion_plus_out_2_constraint.values
            if loc_tech not in
            model._model_data.loc_techs_chp_extraction_cb_constraint
        ]

    if "cv" in model._model_data.data_vars:
        print("Adding chp_cv constraint set")
        model._model_data.coords["loc_techs_chp_extraction_cv_constraint"] = [
            loc_tech
            for loc_tech in model._model_data.loc_techs_conversion_plus.values
            if model._model_data.cv.loc[{"loc_techs": loc_tech}].notnull()
        ]

        print(
            f"{len(model._model_data.loc_techs_chp_extraction_cv_constraint)}"
            " items in set"
        )
        print(
            "Removing conversion_plus loc::techs from "
            "loc_techs_balance_conversion_plus_out_2_constraint"
        )
        model._model_data.coords["loc_techs_balance_conversion_plus_out_2_constraint"] = [
            loc_tech for loc_tech
            in model._model_data.loc_techs_balance_conversion_plus_out_2_constraint.values
            if loc_tech not in
            model._model_data.loc_techs_chp_extraction_cv_constraint
        ]

    if "energy_cap_ratio" in model._model_data.data_vars and "cb" in model._model_data.data_vars:
        print("Adding chp_p2h constraint set")
        model._model_data.coords["loc_techs_chp_extraction_p2h_constraint"] = [
            loc_tech
            for loc_tech in model._model_data.loc_techs_conversion_plus.values
            if model._model_data.cb.loc[{"loc_techs": loc_tech}].notnull()
            and (
                "energy_cap_ratio" in model._model_data.data_vars and
                model._model_data.energy_cap_ratio.loc[{"loc_techs": loc_tech}].notnull()
            )
        ]

        print(
            f"{len(model._model_data.loc_techs_chp_extraction_p2h_constraint)}"
            " items in set"
        )

    if any(var.startswith("capacity_factor") for var in model._model_data.data_vars):
        print("Adding capacity_factor constraint set")

        model._model_data.coords["loc_tech_carriers_capacity_factor_min_constraint"] = [
            loc_tech
            for loc_tech in model._model_data.loc_techs.values
            if model._model_data.capacity_factor_min.loc[{"loc_techs": loc_tech}].notnull()
        ]
        model._model_data.coords["loc_tech_carriers_capacity_factor_max_constraint"] = [
            loc_tech
            for loc_tech in model._model_data.loc_techs.values
            if model._model_data.capacity_factor_max.loc[{"loc_techs": loc_tech}].notnull()
        ]
        print(
            f"{len(model._model_data.loc_tech_carriers_capacity_factor_min_constraint)}"
            " items in set"
        )

    if any(var.startswith("carrier_prod_per_month") for var in model._model_data.data_vars):
        print("Adding carrier_prod_per_month constraint set")
        model._model_data.coords["months"] =np.unique(
            model._model_data.timesteps.dt.month.values
        )

        model._model_data["month_numbers"] = model._model_data.timesteps.dt.month
        model._model_data["month_numbers"].attrs["is_result"] = 0

        for sense in ["min", "equals", "max"]:
            if f"carrier_prod_per_month_{sense}_time_varying" in model._model_data.data_vars:
                model._model_data.coords[
                    f"loc_techs_carrier_prod_per_month_{sense}_constraint"
                ] = [
                    loc_tech
                    for loc_tech in model._model_data.loc_techs.values
                    if (
                        model
                        ._model_data[f"carrier_prod_per_month_{sense}_time_varying"]
                        .loc[{"loc_techs": loc_tech}]
                        .notnull()
                        .all()
                    )
                ]
                print(
                    len(model._model_data[
                        f'loc_techs_carrier_prod_per_month_{sense}_constraint'
                    ]),
                    f" items in {sense} set"
                )
                
    if any(var.startswith("cap_value") for var in model._model_data.data_vars):
        print("Adding capacity_value_constraint set")
        model._model_data.coords["loc_tech_cap_value_constraint"] = [
            loc_techs
            for loc_techs in model._model_data.loc_techs.values
            if (model.inputs.cap_value.loc[{"loc_techs": loc_techs}].notnull().all())
        ]

        print(
            f"{len(model._model_data.loc_tech_cap_value_constraint)}"
            " items in set"
        ) 
        
    return model


def add_eurocalliope_constraints(model):

    if "energy_cap_max_time_varying" in model._model_data.data_vars:
        print("Building production_max_time_varying constraint")
        add_production_max_time_varying_constraint(model)
    if "cb" in model._model_data.data_vars:
        print("Building chp_cb constraint")
        add_chp_cb_constraint(model)
    if "cv" in model._model_data.data_vars:
        print("Building chp_cv constraint")
        add_chp_cv_constraint(model)
    if "energy_cap_ratio" in model._model_data.data_vars and "cb" in model._model_data.data_vars:
        print("Building chp_p2h constraint")
        add_chp_p2h_constraint(model)
    if any(var.startswith("capacity_factor") for var in model._model_data.data_vars):
        print("Building capacity_factor constraint")
        add_capacity_factor_constraints(model)
    if any(var.startswith("carrier_prod_per_month") for var in model._model_data.data_vars):
        print("Building carrier_prod_per_month constraint")
        add_carrier_prod_per_month_constraints(model)
    if any("distribution" in tech for tech in model._model_data.techs.values):
        print("Building fuel distribution constraint")
        add_fuel_distribution_constraint(model)     
    if any("passenger_car_ev_battery" in tech for tech in model._model_data.techs.values):
        print("Building storage_ev_constraint")
        add_storage_ev_constraint(model)
    if any(var.startswith("cap_value") for var in model._model_data.data_vars):
        print("Building cap_value_constraint")
        add_cap_value_constraint(model)

def equalizer(lhs, rhs, sign):
    if sign == "max":
        return lhs <= rhs
    elif sign == "min":
        return lhs >= rhs
    elif sign == "equals":
        return lhs == rhs
    else:
        raise ValueError("Invalid sign: {}".format(sign))

def add_production_max_time_varying_constraint(model):

    def _carrier_production_max_time_varying_constraint_rule(
        backend_model, loc_tech, timestep
    ):
        """
        Set maximum carrier production for technologies with time varying maximum capacity
        """
        energy_cap_max = backend_model.energy_cap_max_time_varying[loc_tech, timestep]

        if invalid(energy_cap_max):
            return po.Constraint.Skip
        model_data_dict = backend_model.__calliope_model_data["data"]
        timestep_resolution = backend_model.timestep_resolution[timestep]
        if loc_tech in backend_model.loc_techs_conversion_plus:
            loc_tech_carriers_out = split_comma_list(
                model_data_dict["lookup_loc_techs_conversion_plus"]["out", loc_tech]
            )
        elif loc_tech in backend_model.loc_techs_conversion:
            loc_tech_carriers_out = [
                model_data_dict["lookup_loc_techs_conversion"]["out", loc_tech]
            ]
        else:
            loc_tech_carriers_out = [
                model_data_dict["lookup_loc_techs"]["out", loc_tech]
            ]

        carrier_prod = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in loc_tech_carriers_out
        )
        return carrier_prod <= (
            backend_model.energy_cap[loc_tech] * timestep_resolution * energy_cap_max
        )

    model.backend.add_constraint(
        "carrier_production_max_time_varying_constraint",
        ["loc_tech_carrier_production_max_time_varying_constraint", "timesteps"],
        _carrier_production_max_time_varying_constraint_rule,
    )


def add_chp_cb_constraint(model):
    def _chp_extraction_cb_constraint_rule(backend_model, loc_tech, timestep):
        """
        Set backpressure line for CHP plants with extraction/condensing turbine
        """
        model_data_dict = backend_model.__calliope_model_data
        loc_tech_carrier_out = model_data_dict["data"]["lookup_loc_techs_conversion_plus"][
            ("out", loc_tech)
        ]
        loc_tech_carrier_out_2 = model_data_dict["data"][
            "lookup_loc_techs_conversion_plus"
        ][("out_2", loc_tech)]

        power_to_heat_ratio = get_param(backend_model, "cb", (loc_tech))

        return backend_model.carrier_prod[loc_tech_carrier_out, timestep] >= (
            backend_model.carrier_prod[loc_tech_carrier_out_2, timestep]
            * power_to_heat_ratio
        )

    model.backend.add_constraint(
        "chp_extraction_cb_constraint",
        ["loc_techs_chp_extraction_cb_constraint", "timesteps"],
        _chp_extraction_cb_constraint_rule,
    )


def add_chp_cv_constraint(model):
    def _chp_extraction_cv_constraint_rule(backend_model, loc_tech, timestep):
        """
        Set extraction line for CHP plants with extraction/condensing turbine
        """
        model_data_dict = backend_model.__calliope_model_data
        loc_tech_carrier_out = model_data_dict["data"]["lookup_loc_techs_conversion_plus"][
            ("out", loc_tech)
        ]
        loc_tech_carrier_out_2 = model_data_dict["data"][
            "lookup_loc_techs_conversion_plus"
        ][("out_2", loc_tech)]

        power_loss_factor = get_param(backend_model, "cv", (loc_tech))

        return backend_model.carrier_prod[loc_tech_carrier_out, timestep] <= (
            backend_model.energy_cap[loc_tech]
            - backend_model.carrier_prod[loc_tech_carrier_out_2, timestep]
            * power_loss_factor
        )

    model.backend.add_constraint(
        "chp_extraction_cv_constraint",
        ["loc_techs_chp_extraction_cv_constraint", "timesteps"],
        _chp_extraction_cv_constraint_rule,
    )


def add_chp_p2h_constraint(model):
    def _chp_extraction_p2h_constraint_rule(backend_model, loc_tech, timestep):
        """
        Set power-to-heat tail for CHPs that allow trading off power output for heat
        """
        model_data_dict = backend_model.__calliope_model_data
        loc_tech_carrier_out = model_data_dict["data"]["lookup_loc_techs_conversion_plus"][
            ("out", loc_tech)
        ]
        loc_tech_carrier_out_2 = model_data_dict["data"][
            "lookup_loc_techs_conversion_plus"
        ][("out_2", loc_tech)]

        power_to_heat_ratio = get_param(backend_model, "cb", loc_tech)
        energy_cap_ratio = get_param(
            backend_model, "energy_cap_ratio", ("out_2", loc_tech_carrier_out_2)
        )
        slope = power_to_heat_ratio / (energy_cap_ratio - 1)
        return backend_model.carrier_prod[loc_tech_carrier_out, timestep] <= (
            slope
            * (
                backend_model.energy_cap[loc_tech] * energy_cap_ratio
                - backend_model.carrier_prod[loc_tech_carrier_out_2, timestep]
            )
        )

    model.backend.add_constraint(
        "chp_extraction_p2h_constraint",
        ["loc_techs_chp_extraction_p2h_constraint", "timesteps"],
        _chp_extraction_p2h_constraint_rule,
    )


def add_capacity_factor_constraints(model):

    def _capacity_factor_min_constraint_rule(backend_model, loc_tech):
        """
        If there is capacity of a technology, force the annual capacity factor to be
        at least a certain amount
        """
        return _capacity_factor_constraint_rule_factory(backend_model, loc_tech, "min")

    def _capacity_factor_max_constraint_rule(backend_model, loc_tech):
        """
        If there is capacity of a technology, force the annual capacity factor to be
        at most a certain amount
        """
        return _capacity_factor_constraint_rule_factory(backend_model, loc_tech, "max")

    def _capacity_factor_constraint_rule_factory(backend_model, loc_tech, sense):
        """
        If there is capacity of a technology, force the annual capacity factor to be
        at most (sense="max") or at least (sense="min") a certain amount
        """
        capacity_factor = get_param(backend_model, f"capacity_factor_{sense}", (loc_tech))
        if invalid(capacity_factor):
            return po.Constraint.Skip
        model_data_dict = backend_model.__calliope_model_data
        loc_tech_carrier = model_data_dict["data"]["lookup_loc_techs"][loc_tech]
        lhs = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            * backend_model.timestep_weights[timestep]
            for timestep in backend_model.timesteps
        )
        rhs = (
            backend_model.energy_cap[loc_tech]
            * capacity_factor
            * get_timestep_weight(backend_model)
            * 8760
        )
        return equalizer(lhs, rhs, sense)

    model.backend.add_constraint(
        "capacity_factor_min_constraint",
        ["loc_tech_carriers_capacity_factor_min_constraint"],
        _capacity_factor_min_constraint_rule,
    )
    model.backend.add_constraint(
        "capacity_factor_max_constraint",
        ["loc_tech_carriers_capacity_factor_max_constraint"],
        _capacity_factor_max_constraint_rule,
    )


def add_carrier_prod_per_month_constraints(model):

    def _carrier_prod_per_month_constraint_rule_generator(sense):
        def __carrier_prod_per_month_constraint_rule(backend_model, loc_tech, month):
            """
            Set the min/max amount of carrier consumption (relative to annual consumption)
            for a specific loc tech that must take place in a given calender month in the model
            """
            model_data_dict = backend_model.__calliope_model_data
            loc_tech_carrier = model_data_dict["data"]["lookup_loc_techs_conversion"][("out", loc_tech)]

            prod = backend_model.carrier_prod
            prod_total = sum(
                prod[loc_tech_carrier, timestep]
                for timestep in backend_model.timesteps
            )
            prod_month = sum(
                prod[loc_tech_carrier, timestep]
                for timestep in backend_model.timesteps
                if backend_model.month_numbers[timestep].value == month
            )
            if "timesteps" in [
                i.name
                for i in getattr(
                    backend_model, f"carrier_prod_per_month_{sense}_time_varying"
                )._index.subsets()
            ]:
                prod_fraction = sum(
                    get_param(
                        backend_model,
                        f"carrier_prod_per_month_{sense}_time_varying",
                        (loc_tech, timestep),
                    )
                    * backend_model.timestep_resolution[timestep]
                    for timestep in backend_model.timesteps
                    if backend_model.month_numbers[timestep].value == month
                )
            else:
                prod_fraction = get_param(
                    backend_model, f"carrier_prod_per_month_{sense}", (loc_tech)
                )

            return equalizer(prod_month, prod_total * prod_fraction, sense)
        return __carrier_prod_per_month_constraint_rule


    for sense in ["min", "max", "equals"]:
        if f"carrier_prod_per_month_{sense}_time_varying" in model._model_data.data_vars:
            model.backend.add_constraint(
                f"carrier_prod_per_month_{sense}_constraint",
                [f"loc_techs_carrier_prod_per_month_{sense}_constraint", "months"],
                _carrier_prod_per_month_constraint_rule_generator(sense),
            )


def add_fuel_distribution_constraint(model):
    def _fuel_distribution_constraint_rule(backend_model, carrier):
        tech_import = f"{carrier}_distribution_import"
        tech_export = f"{carrier}_distribution_export"
        if tech_import not in backend_model.techs:
            return po.Constraint.Skip

        tech_carrier_import = f"{tech_import}::{carrier}"
        tech_carrier_export = f"{tech_export}::{carrier}"
        carrier_import = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for timestep in backend_model.timesteps
            for loc_tech_carrier in backend_model.loc_tech_carriers_prod
            if tech_carrier_import in loc_tech_carrier
        )
        carrier_export = sum(
            backend_model.carrier_con[loc_tech_carrier, timestep]
            for timestep in backend_model.timesteps
            for loc_tech_carrier in backend_model.loc_tech_carriers_con
            if tech_carrier_export in loc_tech_carrier
        )
        return carrier_import == -1 * carrier_export

    model.backend.add_constraint(
        "fuel_distribution_constraint",
        ["carriers"],
        _fuel_distribution_constraint_rule,
    )


def add_storage_ev_constraint(model):
    def _storage_ev_constraint_rule(backend_model, loc_tech):
        share_loc = loc_tech.rsplit("::")[0]
        share_tech_carrier = "passenger_car_transport_ev::passenger_car_transport"      
        loc_tech_carrier = f"{share_loc}::{share_tech_carrier}"
        storage_cap_max = get_param(backend_model, "storage_cap_max", (loc_tech) )
                
        try:
            backend_model.demand_share_per_timestep_decision[loc_tech_carrier].value
        except KeyError:
            return po.Constraint.Skip
        # print(loc_tech_carrier)  
        return backend_model.storage_cap[loc_tech] <= ( backend_model.demand_share_per_timestep_decision[loc_tech_carrier] * storage_cap_max )
                
    model.backend.add_constraint(
        "storage_ev_constraint",
        ["loc_tech_ev_storage_constraint"],
        _storage_ev_constraint_rule
    )

def add_cap_value_constraint(model):
    def _cap_value_rule(backend_model, loc_tech, timestep):
        
        cap_value = get_param(backend_model, "cap_value", (loc_tech,timestep))
        # print(cap_value)
        
        if invalid(cap_value):
            return po.Constraint.Skip
        
        loc_tech_carrier = f"{loc_tech}::electricity"
        carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep]
        timestep_resolution = backend_model.timestep_resolution[timestep]  
        
        return carrier_prod <= (
            backend_model.energy_cap[loc_tech] * timestep_resolution * cap_value
        )
        
    model.backend.add_constraint(
        "cap_value_constraint",
        ["loc_tech_cap_value_constraint", "timesteps"],
        _cap_value_rule
    )

def add_target_reserve_share_constraint(model):
    def _target_reserve_share_rule(backend_model, group_name, what):
        """
        Enforces carrier_prod for groups of technologies and locations,
        as a sum over the entire model period. UPDATE
    
        .. container:: scrolling-wrapper
    
            .. math:: UPDATE
    
                \\sum_{loc::tech::carrier \\in given\\_group, timestep \\in timesteps} carrier_{prod}(loc::tech::carrier, timestep) \\leq carrier_prod_max
    
        """
        limit = get_param(backend_model, f"group_target_reserve_share_{what}", (group_name))
    
        if invalid(limit):
            return return_noconstraint("target_reserve_share", group_name)
        else:
            lhs_loc_tech_carriers = get_group_lhs_loc_tech_carriers(
                backend_model, group_name
            )
    
            lhs = sum(
                backend_model.carrier_prod[loc_tech_carrier, timestep]
                for loc_tech_carrier in lhs_loc_tech_carriers
                for timestep in backend_model.timesteps
            )
            return equalizer(lhs, limit, what)

        model.backend.add_constraint(
            "target_reserve_share_constraint",
            ["loc_tech_cap_value_constraint", "timesteps"],
            _target_reserve_share_rule
        )

def return_noconstraint(*args):
    logger = logging.getLogger(__name__)
    logger.debug("group constraint returned NoConstraint: {}".format(",".join(args)))
    return po.Constraint.NoConstraint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model_nc", "-i", help="Path to built model inputs NetCDF file", type=str)
    parser.add_argument("output_model_nc", "-o", help="Path to optimised model NetCDF file", type=str)

    args = parser.parse_args()

    run_model(args.input_model_nc, args.output_model_nc)
