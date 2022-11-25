# Import packages
import numpy as np
import torch # device assignment for importance model

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
from modulus.constraint import Constraint # don't know
from modulus.continuous.validator.validator import PointwiseValidator  # adding validation dataset
from modulus.continuous.inferencer.inferencer import PointwiseInferencer # infer variables
from modulus.tensorboard_utils.plotter import ValidatorPlotter, InferencerPlotter # tensorboard
from modulus.key import Key  # add keys to nodes
from modulus.node import Node # add node to geometry

from modulus.pdes import PDES # PDEs
from modulus.graph import Graph # for defining importance graph model
from itertools import product # create combinations of elements of multiple arrays

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

    # Define parametric PDE inputs
    if cfg.custom.parameterized:
        input_keys = [Key("x"), Key("y"), Key("L")] # one more input
    else:
        input_keys=[Key("x"), Key("y")]
    
    # make list of nodes to unroll graph on
    heat_eq = HeatConductionEquation2D()
    heat_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("T")], # output keys remains the same
        cfg=cfg.arch.fully_connected,
    )
    
    nodes = heat_eq.make_nodes() + [heat_net.make_node(name="heat_network", jit=cfg.jit)]
    print('Nodes Initialised')
    
    # make importance model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    
    importance_model_graph = Graph(
        nodes,
        invar=[Key("x"), Key("y"), Key("L")],
        req_names=[
            Key("T", derivatives=[Key("x")]),
            Key("T", derivatives=[Key("y")]),
        ],
    ).to(device)

    def importance_measure(invar):
        outvar = importance_model_graph(
            Constraint._set_device(invar, requires_grad=True)
        )
        importance = (
            outvar["T__x"] ** 2
            + outvar["T__y"] ** 2
        ) ** 0.5 + 10
        return importance.cpu().detach().numpy()

    
    # make geometry
    x, y, L = Symbol("x"), Symbol("y"), Symbol('L')
    lower_bound = (-0.5, 0)
    upper_bound = (0.5, L) # y - upper bound is parametric 
    
    
    rec = Rectangle(lower_bound, upper_bound) # (x_1,y_1), (x_2,y_2)
    pr = {Symbol("L"): (1, 1.05)} # param_ranges for variation of parametric L
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
        importance_measure=importance_measure,
        param_ranges=pr,
        
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
        criteria=Eq(y, L), # coordinates for sampling
        importance_measure=importance_measure,
        param_ranges=pr,
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
        importance_measure=importance_measure,
        param_ranges=pr,
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
        importance_measure=importance_measure,
        param_ranges=pr,
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
        bounds={x: x_bound, y: (0,1.05)}, # sampling should be from along y 0,10. see baseline PINN version for some intuition
        # lambda weights for now
        lambda_weighting={"heat_equation": rec.sdf},
        importance_measure=importance_measure,
        param_ranges=pr,
    )
    heat_domain.add_constraint(interior, "interior")
    
    print('Interior points created')
    
#     # Getting path (nvidia modulus only accepts absolute path to remove any ambiguity)
#     validation_path = to_absolute_path("final_data.npz")

#     # Adding VALIDATION data
#     data = np.load(validation_path) # Load data from file
#     keys = list(data.keys()) # all keys in the dictionary
#     print('Validation dataset keys ',keys)
#     nodes3D = data[keys[0]]
#     temperature = data[keys[1]]
#     boundary_nodal_coordinates3D = data[keys[2]]
#     boundary_solution = data[keys[3]]
#     # face3 = data[keys[4]]
#     # face4 = data[keys[5]]
#     # face5 = data[keys[6]]
#     # face6 = data[keys[7]]
    
    
#     # cutting the useless third dimension where there is nothing to predict
#     node = nodes3D[:,[0,2]] # remember there is nodes variable also, they should not clash
#     #temperature = temperature3D[:,[0,2]]
#     boundary_nodal_coordinates = boundary_nodal_coordinates3D[:,[0,2]]
#     #boundary_solution = boundary_solution3D[:,[0,2]]
    
    
#     # This is the required format for validation data
#     openfoam_invar_numpy = {}
#     openfoam_invar_numpy["x"] = node[:, 0][:, None]
#     openfoam_invar_numpy["y"] = node[:, 1][:, None]

#     openfoam_outvar_numpy = {}
#     openfoam_outvar_numpy["T"] = temperature
    
#     openfoam_validator = PointwiseValidator(
#         openfoam_invar_numpy,
#         openfoam_outvar_numpy,
#         nodes,
#         batch_size=1024,
#         plotter=ValidatorPlotter(),
#     )
#     heat_domain.add_validator(openfoam_validator)
    
    print('Validation Data loaded')
    
    training_dataset = np.array([]) # the training dataset
    L_ranges = np.linspace(1, 1.05, 6) # the 3rd input
    x_ranges = np.linspace(-0.5, 0.5, 80) # x coordinates

    for length_parameter in L_ranges: # for each length param
        # y is used to satisfy the PDE i.e the collocation points
        y_ranges = np.linspace(0, length_parameter, 80) # y coodinates should be less the length_parameter i.e. inside the variable upper boundary

        temp_dataset = np.array(list(product(x_ranges, y_ranges, [length_parameter]))) # create an numpy from itertools.product

        training_dataset = np.vstack([training_dataset, temp_dataset]) if training_dataset.size else temp_dataset
    
    invar_numpy = {"x": training_dataset[:,0][:,None], "y": training_dataset[:,1][:,None], "L": training_dataset[:,2][:,None]}
    # add INFERENCER data
    grid_inference = PointwiseInferencer(
        invar_numpy,
        ["T"],#, "T__x", "T__y"],
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
