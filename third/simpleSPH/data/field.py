import numpy as np

coords            = "coords"
velocity          = "velocity"
density           = "density"
densityDerivative = "densityDerivative"
size              = "size"
element           = "element"
material          = "material"
mass              = "mass"
energy            = "energy"
innerEnergy       = "innerEnergy"
pressure          = "pressure"
momentum          = "momentum"
soundSpeed        = "soundSpeed"
force             = "force"
energyDerivative  = "energyDerivative"
strainRate        = "strainRate"
left              = "left"
right             = "right"
neibs             = "neibs"
# for tests
coordsSolution    = "coordsSolution"
densitySolution   = "densitySolution"
velocitySolution  = "velocitySolution"
pressureSolution  = "pressureSolution"
energySolution    = "energySolution"

fields_dtype = {
    coords:            np.float64,
    velocity:          np.float64,
    density:           np.float64,
    densityDerivative: np.float64,
    size:              np.float64,
    element:           "20S",
    material:          np.int64,
    mass:              np.float64,
    energy:            np.float64,
    innerEnergy:       np.float64,
    pressure:          np.float64,
    momentum:          np.float64,
    soundSpeed:        np.float64,
    force:             np.float64,
    energyDerivative:  np.float64,
    strainRate:        np.float64,
    left:              np.int64,
    right:             np.int64,
    neibs:             np.int64
}

fields_list = [
    coords,
    velocity,
    density,
    densityDerivative,
    size,
    element,
    material,
    mass,
    energy,
    innerEnergy,
    pressure,
    momentum,
    soundSpeed,
    force,
    energyDerivative,
    strainRate,
    left,
    right,
    neibs
]