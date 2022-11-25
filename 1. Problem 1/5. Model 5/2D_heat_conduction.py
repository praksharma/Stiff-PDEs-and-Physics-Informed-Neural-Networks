# Import packages
import numpy as np
from sympy import Symbol, Eq, Function, Number, Abs # defining symbols

import modulus # load classes and functions from modulus directory
from modulus.hydra import to_absolute_path, to_yaml, instantiate_arch  # configure settings and create architecture
from modulus.hydra.config import ModulusConfig # load config files
#from modulus.csv_utils.csv_rw import csv_to_dict  # load true solution for validation
from modulus.continuous.solvers.solver import Solver # continuous time solver
from modulus.continuous.domain.domain import Domain  # domain (custom classes + architecture)
from modulus.geometry.csg.csg_2d import Rectangle  # CSG geometry
from modulus.continuous.constraints.constraint import (  # constraints
    PointwiseBoundaryConstraint,  # BC
    PointwiseInteriorConstraint,  # interior/collocation points
)
from modulus.continuous.validator.validator import PointwiseValidator  # adding validation dataset
from modulus.continuous.inferencer.inferencer import PointwiseInferencer # infer variables
from modulus.tensorboard_utils.plotter import ValidatorPlotter, InferencerPlotter # tensorboard
from modulus.key import Key  # add keys to nodes
from modulus.node import Node # add node to geometry

from modulus.pdes import PDES # PDEs


class HeatConductionEquation2D(PDES):
    """
    Heat Conduction 2D
    
    Parameters
    ==========
    k : float, string
        Conductivity doesn't matter in steady state problem with no heat source
    """

    name = "HeatConductionEquation2D"
    def __init__(self):
        # coordinates
        x = Symbol("x")

        # time
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # Temperature output
        T = Function("T")(*input_variables)

        # conductivity coefficient
        # if type(c) is str:
        #     c = Function(c)(*input_variables) # if c is function of independent variables
        # elif type(c) in [float, int]:
        #     c = Number(c)

        # set equations
        self.equations = {}
        self.equations["heat_equation"] = T.diff(x, 2) + T.diff(y, 2)  # diff(variable,order of derivative)

        
@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    # make list of nodes to unroll graph on
    heat_eq = HeatConductionEquation2D()
    heat_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("T")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = heat_eq.make_nodes() + [heat_net.make_node(name="heat_network", jit=cfg.jit)]
    print('Nodes Initialised')
    
    # make geometry
    lower_bound = (-0.5, 0)
    upper_bound = (0.5, 1)
    
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle(lower_bound, upper_bound) # (x_1,y_1), (x_2,y_2)
    
    print('Geometry created')
          
    # make domain
    heat_domain = Domain()
    print('Domain created')
          
    # Adding constraints
    # Bottom wall
    bottom_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 0.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        lambda_weighting={"T": 1.0 - 2 * Abs(x)},  # weight edges to be zero
        criteria=Eq(y, 0.0), # coordinates for sampling
    )
    heat_domain.add_constraint(bottom_wall, "bottom_wall")
    
    print('Bottom wall BC created')
          
    # Top wall
    top_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 0.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        lambda_weighting={"T": 1.0 - 2 * Abs(x)},  # weight edges to be zero
        criteria=Eq(y, 1.0), # coordinates for sampling
    )
    heat_domain.add_constraint(top_wall, "top_wall")

    print('Top wall BC created')
    
    # Left wall
    left_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 1.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, -0.5), # coordinates for sampling
    )
    heat_domain.add_constraint(left_wall, "left_wall")
    
    print('Left wall BC created')
          
    # Right wall
    right_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 1.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, 0.5), # coordinates for sampling
    )
    heat_domain.add_constraint(right_wall, "right_wall")
    
    print('Right wall BC created')

    # PDE constraint
    x_bound = (-0.5,0.5)
    y_bound = (0, 1)
    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"heat_equation" : 0},
        batch_size=cfg.batch_size.Interior,
        bounds={x: x_bound, y: y_bound},
        # No lambda weights for now
        lambda_weighting={"heat_equation": rec.sdf},
    )
    heat_domain.add_constraint(interior, "interior")
    
    print('Interior points created')
    
    # Getting path (nvidia modulus only accepts absolute path to remove any ambiguity)
    validation_path = to_absolute_path("final_data.npz")

    # Adding VALIDATION data
    data = np.load(validation_path) # Load data from file    keys = list(data.keys()) # all keys in the dictionary
    keys = list(data.keys()) # all keys in the dictionary
    print('Validation dataset keys ',keys)
    nodes3D = data[keys[0]]
    temperature = data[keys[1]]
    boundary_nodal_coordinates3D = data[keys[2]]
    boundary_solution = data[keys[3]]
    # face3 = data[keys[4]]
    # face4 = data[keys[5]]
    # face5 = data[keys[6]]
    # face6 = data[keys[7]]
    
    
    # cutting the useless third dimension where there is nothing to predict
    node = nodes3D[:,[0,2]] # remember there is nodes variable also, they should not clash
    #temperature = temperature3D[:,[0,2]]
    boundary_nodal_coordinates = boundary_nodal_coordinates3D[:,[0,2]]
    #boundary_solution = boundary_solution3D[:,[0,2]]
    
    
    # This is the required format for validation data
    openfoam_invar_numpy = {}
    openfoam_invar_numpy["x"] = node[:, 0][:, None]
    openfoam_invar_numpy["y"] = node[:, 1][:, None]

    openfoam_outvar_numpy = {}
    openfoam_outvar_numpy["T"] = temperature
    
    openfoam_validator = PointwiseValidator(
        openfoam_invar_numpy,
        openfoam_outvar_numpy,
        nodes,
        batch_size=1024,
        plotter=ValidatorPlotter(),
    )
    heat_domain.add_validator(openfoam_validator)
    
    print('Validation Data loaded')
    
    # add INFERENCER data
    grid_inference = PointwiseInferencer(
        openfoam_invar_numpy,
        ["T"],
        nodes,
        batch_size=1024,
        plotter=InferencerPlotter(),
    )
    heat_domain.add_inferencer(grid_inference, "inf_data")
    
    print('Inference outputs loaded')
    
        # make solver
    slv = Solver(cfg, heat_domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
        
