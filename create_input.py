import argparse

import calliope


def build_model(path_to_model, scenario, path_to_output):

    calliope.set_log_verbosity("info", include_solver_output=True, capture_warnings=True)
    model = calliope.Model(path_to_model, scenario=scenario)

    model._model_data.attrs["scenario"] = scenario

    model.to_netcdf(path_to_output)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model_yaml", "-i", help="Path to model YAML file", type=str)
    parser.add_argument("output_model_nc", "-o", help="Path to built model NetCDF file", type=str)
    parser.add_argument("--scenario", help="comma separated list of scenario or override names", default=None, type=str)

    args = parser.parse_args()

    build_model(args.input_model_yaml, args.scenario, args.output_model_nc)
