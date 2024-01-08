### Introduction ###

This repo is intended to test and further dig into the effects of introducing newly implemented reserve margins on the modelled energy systems.
The formulation of the reserve margins is hard-coded into a branch of Calliope [(here)](https://github.com/FraSanvit/calliope/tree/0.6-reserves-margins).

If you want to contribute consider the [PR](https://github.com/calliope-project/calliope/pull/517) in the Calliope project repository.

### Features of reserve margins ###

Two categories of reserve margins are implemented, namely: (i) planning reserve and (ii) operating reserve. Each type has different modes depending on how the requirements are defined.

1. Planning reserve
    * Percentage of the system peak
    * Absolute adder to the system peak
    * Absolute value
3. Operating reserve
    * Percentage of the net load
    * Absolute adder
    * Absolute value

<p align="center">
<img src="https://github.com/FraSanvit/reserve-margins/blob/main/docs/new_group_constraint_names.png" width="600">
</p>
  
Along these modes, two extra constraint features have been introduced:
1. `capacity value` or derating factor (for both planning and operating reserves)
2. `operating reserve` targets or generation/capacity target additions (only in the operating reserve)

<p align="center">
<img src="https://github.com/FraSanvit/reserve-margins/blob/main/docs/new_constraint_names.png" width="600">
</p>

### Reserve margins in Calliope ###

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
As already mentioned, `cap_value` can be also a constant value of the time horizon.

Likewise, the `operating_reserve` can be a timeseries or a constant value and it represents the additional requirement due to the capacity or generation values of specific technologies. It is usually defined as a share of the installed capacity or generation.
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
            operating_reserve: file=operating-reserve-open-field-pv.csv
```

The reserve margins are implemented as group constraints and they do follow the same structure. The new feature of reserve margin group constraints consists of the possibility to input timeseries as targets (in the operating mode only).

**Planning reserve - percentage of the system peak**

The group constraint applies to a specific carrier and the value refers to the additional share of the target.

```
group_constraints:
    reserve_margin_1:
        techs: [ccgt,open_field_pv]
        locs: [N1]
        target_reserve_share_min:
            electricity: 0.50
```
In the given example, the target reserve requirement is set to be an additional +50% of the installed capacity, compared to what would have been necessary if reserve margins were not implemented.

**Planning reserve - absolute adder to the system peak**

The target reserve is equal to the capacity required by the system if reserve margins were not implemented plus an adder value expressed in capacity units.
```
group_constraints:
    reserve_margin_1:
        techs: [ccgt,open_field_pv]
        locs: [N1]
        target_reserve_adder_min:
            electricity: 50 # (MW)
```

**Planning reserve - absolute value**

The target reserve is equal to an absolute value expressed in capacity units. This constraint might look similar to setting a minimum value of capcity deployment but indeed it includes also the effect of the capacity values.
```
group_constraints:
    reserve_margin_1:
        techs: [ccgt, open_field_pv]
        locs: [N1]
        target_reserve_abs_min:
            electricity: 1000 # (MW)
```

**Operating reserve - percentage of the net load**

All operating targets are composed by an additional term, besides the load, that consists of extra requirements expressed as a share of generating capacity or energy production. This terms is the one introduced in the new constraint `operating_reserve`. The operatign reserve can be defined for each technology and location.

In this specific case, the load is multiplied by a factor (1 + `% of the net laod`) where the `% of the net laod` is introduced as follows:
```
    reserve_margin_1:
        techs: [ccgt, open_field_pv]
        locs: [N1]
        target_reserve_share_operating_min:
            electricity: file=operating-reserve-target.csv:reserve_margin_1 # (%)
```

**Operating reserve - absolute adder**

In the absolute adder mode, the target reserve is increased by an amount of capacity expressed in capacity units.
```
    reserve_margin_1:
        techs: [ccgt, open_field_pv]
        locs: [N1]
        target_reserve_adder_operating_min:
            electricity: file=operating-reserve-target.csv:reserve_margin_1 # (MW)
```

**Operating reserve - absolute value**

In the absolute mode, the target reserve is equal to a fixed capacity amount expressed in capacity units.
```
    reserve_margin_1:
        techs: [ccgt, open_field_pv]
        locs: [N1]
        target_reserve_adder_operating_min:
            electricity: file=operating-reserve-target.csv:reserve_margin_1 # (MW)
```

**[WARNING]:** When you use timeseries inputs for operating reserve targets, you must provide the specific column of the file. The column has to reflect the name of the group constraint.
