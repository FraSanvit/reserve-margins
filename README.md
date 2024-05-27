### Introduction ###

This repository is designed to test and delve deeper into the impacts of introducing newly implemented reserve margins on modeled energy systems. The formulation of the reserve margins is hardcoded into a branch of Calliope, accessible [(here)](https://github.com/FraSanvit/calliope/tree/0.6-reserves-margins-fix).

### Features of reserve margins ###

Two categories of reserve margins are implemented, namely: (i) planning reserve and (ii) operating reserve. Each type has different modes based on how the requirements are defined.

1. Planning reserve
    * Percentage of the system peak
    * Absolute adder to the system peak
    * Absolute value
2. Operating reserve
    * Percentage of the net load
    * Absolute adder
    * Absolute value

<p align="center">
<img src="https://github.com/FraSanvit/reserve-margins/blob/main/docs/new_group_constraint_names.png" width="600">
</p>

The operating reserve are further characterised by reserve types which are: `frequency` (freq), `flexibility` (flex), `contingency` (cont) and `regulation` (reg).

Along these modes, two extra constraint features are introduced:
1. `capacity value` or derating factor (for both planning and operating reserves)
2. operating reserve targets additions. The operating reserve target additions are expressed in function of the production or installed capacity as follows:
   * `operating reserve` (operating_reserve)
   * `operating reserve capacity` (operating_reserve_cap)

<p align="center">
<img src="https://github.com/FraSanvit/reserve-margins/blob/main/docs/new_constraint_names.png" width="600">
</p>

First of all, the derating factors applied to the capacities can be defined in the following way:
```
techs:
    ccgt:
        essentials:
            name: Combined cycle gas turbine
            carrier_out: electricity
            carrier_in: methane
            parent: conversion
        constraints:
            energy_eff: 0.53
            lifetime: 25
            cap_value: file=cap-values-ccgt.csv
```
As mentioned earlier, `cap_value` can also be a constant value for the entire time horizon.

Similarly, `operating_reserve` and `operating_reserve_cap` can be either a time series or a constant value, representing the additional requirement due to the capacity or generation values of specific technologies.
```
techs:
    open_field_pv:
        essentials:
            name: Open field PV
            parent: pv
        constraints:
            resource_area_per_energy_cap: 0.125 # (0.1 km^2/MW)
            resource: file=capacityfactors-open-field-pv.csv
            resource_unit: energy_per_cap
            operating_reserve_cap: file=operating-reserve-open-field-pv.csv
```

The reserve margins are implemented as group constraints and follow the same structure. The new feature of reserve margin group constraints includes the option to input timeseries as targets.

### Planning reserves ###

In the planning reserve the target value included in the group constraint cannot be loaded as a timeseries since the reserve margin applies by default to the system peak demand timestep.

**Percentage of the system peak**

The group constraint applies to a specific carrier and the value refers to the additional share of the target.

```
group_constraints:
    reserve_margin_1:
        techs: [ccgt,open_field_pv]
        locs: [N1]
        target_reserve_share_min:
            electricity: 0.50
```
In this example, the target reserve requirement is defined to be an extra +50% of the installed capacity. This is in addition to what would have been required if reserve margins were not implemented.

**Absolute adder to the system peak**

The target reserve is equal to the capacity required by the system if reserve margins were not implemented plus an adder value expressed in capacity units.
```
group_constraints:
    reserve_margin_1:
        techs: [ccgt,open_field_pv]
        locs: [N1]
        target_reserve_adder_min:
            electricity: 50 # (MW)
```

**Absolute value**

The target reserve is specified as an absolute value, expressed in capacity units. While this constraint may resemble setting a minimum capacity deployment, it actually encompasses the impact of both the capacity values and the specified absolute value.
```
group_constraints:
    reserve_margin_1:
        techs: [ccgt, open_field_pv]
        locs: [N1]
        target_reserve_abs_min:
            electricity: 1000 # (MW)
```

### Operating reserves ###

**Percentage of the net load**

All operating targets consist of an additional term, in addition to the load, representing extra requirements expressed as a proportion of generating capacity or energy production. This term is introduced through the new constraint `operating_reserve` and `operating_reserve_cap`, which can be defined for each technology and location.

In this specific case, the load is multiplied by a factor of (1 + `% of the net laod`) where the `% of the net laod` is introduced as follows:
```
group_constraints:
    reserve_margin_1:
        techs: [ccgt, open_field_pv]
        locs: [N1]
        target_reserve_share_operating_{$reserve_type$}_{$sense$}:
            electricity: file=operating-reserve-target.csv:reserve_margin_1 # (%)
```
Constraints for {$reserve_type$}:
   * freq
   * flex
   * cont
   * reg

Constraints for {$sense$}:
   * min
   * max
   * equals

**Absolute adder**

In the absolute adder mode, the target reserve is increased by an amount of capacity expressed in capacity units.
```
group_constraints:
    reserve_margin_1:
        techs: [ccgt, open_field_pv]
        locs: [N1]
        target_reserve_adder_operating_{$reserve_type$}_{$sense$}:
            electricity: file=operating-reserve-target.csv:reserve_margin_1 # (MW)
```

**Absolute value**

In the absolute mode, the target reserve is equal to a fixed capacity amount expressed in capacity units.
```
group_constraints:
    reserve_margin_1:
        techs: [ccgt, open_field_pv]
        locs: [N1]
        target_reserve_adder_operating_{$reserve_type$}_{$sense$}:
            electricity: file=operating-reserve-target.csv:reserve_margin_1 # (MW)
```

**[WARNING]:** When using timeseries inputs for operating reserve targets, you need to specify the particular column in the file. The column must correspond to the name of the group constraint.
