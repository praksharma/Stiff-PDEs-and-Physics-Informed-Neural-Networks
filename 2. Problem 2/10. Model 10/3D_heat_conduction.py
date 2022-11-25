# Import packages
import numpy as np
from sympy import Symbol, Eq, Function, Number, Abs # defining symbols
import torch # device assignment for importance model

import modulus # load classes and functions from modulus directory
from modulus.hydra import to_absolute_path, to_yaml, instantiate_arch  # configure settings and create architecture
from modulus.hydra.config import ModulusConfig # load config files
#from modulus.csv_utils.csv_rw import csv_to_dict  # load true solution for validation
from modulus.continuous.solvers.solver import Solver # continuous time solver
from modulus.continuous.domain.domain import Domain  # domain (custom classes + architecture)
from modulus.geometry.csg.csg_3d import Box  # CSG geometry
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

from modulus.architecture.fully_connected import FullyConnectedArch
from modulus.architecture.fourier_net import FourierNetArch
from modulus.architecture.siren import SirenArch
from modulus.architecture.modified_fourier_net import ModifiedFourierNetArch
from modulus.architecture.dgm import DGMArch


class HeatConductionEquation3D(PDES):
    """
    Heat Conduction 2D
    
    Parameters
    ==========
    k : float, string
        Conductivity doesn't matter in steady state problem with no heat source
    """

    name = "HeatConductionEquation3D"
    def __init__(self):
        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # Temperature output
        T = Function("T")(*input_variables)

        # conductivity coefficient
        # if type(c) is str:
        #     c = Function(c)(*input_variables) # if c is function of independent variables
        # elif type(c) in [float, int]:
        #     c = Number(c)

        # set equations
        self.equations = {}
        self.equations["heat_equation"] = T.diff(x, 2) + T.diff(y, 2) +  T.diff(z, 2) # diff(variable,order of derivative)
        
        
@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    input_keys = [Key("x"), Key("y"), Key("z")]
    output_keys=[Key("T")]
    
    # make list of nodes to unroll graph on
    heat_eq = HeatConductionEquation3D()
    if cfg.custom.arch == "FullyConnectedArch":
        heat_net = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "FourierNetArch":
        print('Architecture: Fourier Network')
        heat_net = FourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "SirenArch":
        print('Architecture: SIREN Arch')
        heat_net = SirenArch(
            input_keys=input_keys,
            output_keys=output_keys,
            normalization={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0, 1.0)}, # no idea
        )
    elif cfg.custom.arch == "ModifiedFourierNetArch":
        heat_net = ModifiedFourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    else:
        sys.exit(
            "Network not configured for this script. Please include the network in the script"
        )
    
    
    
    # heat_net = instantiate_arch(
    #     input_keys=input_keys,
    #     output_keys=output_keys,
    #     cfg=cfg.arch.fully_connected,
    # )
    nodes = heat_eq.make_nodes() + [heat_net.make_node(name="heat_network", jit=cfg.jit)]
    print('Nodes Initialised')
    
    # make importance model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    
    importance_model_graph = Graph(
        nodes,
        invar=[Key("x"), Key("y"), Key("z")],
        req_names=[
            Key("T", derivatives=[Key("x")]),
            Key("T", derivatives=[Key("y")]),
            Key("T", derivatives=[Key("z")]),
        ],
    ).to(device)

    def importance_measure(invar):
        outvar = importance_model_graph(
            Constraint._set_device(invar, requires_grad=True)
        )
        importance = (
            outvar["T__x"] ** 2
            + outvar["T__y"] ** 2
            + outvar["T__z"] ** 2
        ) ** 0.5 + 10
        return importance.cpu().detach().numpy()

    
    
    # make geometry
    lower_bound = (-0.5, -0.5, 0.0) #[[-0.5, -0.5, 0], [0.5, 0.5, 1]]
    upper_bound = (0.5, 0.5, 1.0)
        
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    rec = Box(lower_bound, upper_bound) # (x_1,y_1,z_1), (x_2,y_2,z_2) # defining cuboid
    
    print('Geometry created')
          
    # make domain
    heat_domain = Domain()
    print('Domain created')
          
    # Adding constraints (in order of constraints defined in cell[1])
    # wall 1 : u(x,y,0) = 0 
    wall_1 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 0.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(x)},  # weight edges to be zero
        criteria=Eq(z, 0.0), # coordinates for sampling
        importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(wall_1, "wall_1")
    
    print('wall_1 BC created')
          
    # wall_2 : u(x,y,1) = 0 
    wall_2 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 0.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(x)},  # weight edges to be zero
        criteria=Eq(z, 1.0), # coordinates for sampling
        importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(wall_2, "wall_2")

    print('wall_2 BC created')
    
    # wall_3 : u(-0.5,y,z) = 1
    wall_3 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 1.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, -0.5), # coordinates for sampling
        importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(wall_3, "wall_3")
    
    print('wall_3 BC created')
          
    # wall_4 : u(0.5,y,z) = 0
    wall_4 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 0.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, 0.5), # coordinates for sampling
        importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(wall_4, "wall_4")
    
    print('wall_4 BC created')
    
    # wall_5 : u(x,-0.5,z) = 1
    wall_5 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 1.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(y, -0.5), # coordinates for sampling
        importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(wall_5, "wall_5")
    
    print('wall_5 BC created')
    
    # wall_6 : u(x,0.5,z) = 1
    wall_6 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 1.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(y, 0.5), # coordinates for sampling
        importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(wall_6, "wall_6")
    
    print('wall_6 BC created')

    # PDE constraint
    x_bound = (-0.5,0.5)
    y_bound = (-0.5,0.5)
    z_bound = (0.0,1.0)
    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"heat_equation" : 0.0},
        batch_size=cfg.batch_size.Interior,
        bounds={x: x_bound, y: y_bound, z: z_bound},
        # lambda weights for now
        lambda_weighting={"heat_equation": rec.sdf},
        importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(interior, "interior")
    
    print('Interior points created')
    
    # Getting path (nvidia modulus only accepts absolute path to remove any ambiguity)
    validation_path = to_absolute_path("new_final_data.npz")
    print(validation_path)

    # Loading validation data
    data = np.load(validation_path) # Load data from file
    keys = list(data.keys()) # all keys in the dictionary
    print(keys)
    node = data[keys[0]]
    temperature = data[keys[1]]
    #boundary_nodal_coordinates3D = data[keys[2]]
    #boundary_solution = data[keys[3]]
    # face3 = data[keys[4]]
    # face4 = data[keys[5]]
    # face5 = data[keys[6]]
    # face6 = data[keys[7]]
    #print(np.shape(node))
    #print(np.shape(temperature))
    #print(np.shape(boundary_nodal_coordinates3D))
    #print(np.shape(boundary_solution))
    
    
    # cutting the useless third dimension where there is nothing to predict
    #node = nodes3D[:,[0,2]] # remember there is nodes variable also, they should not clash
    #temperature = temperature3D[:,[0,2]]
    #boundary_nodal_coordinates = boundary_nodal_coordinates3D[:,[0,2]]
    #boundary_solution = boundary_solution3D[:,[0,2]]
    
    
    # This is the required format for validation data
    openfoam_invar_numpy = {}
    openfoam_invar_numpy["x"] = node[:, 0][:, None]
    openfoam_invar_numpy["y"] = node[:, 1][:, None]
    openfoam_invar_numpy["z"] = node[:, 2][:, None]

    openfoam_outvar_numpy = {}
    # the file has T from 100 to 200
    openfoam_outvar_numpy["T"] = temperature/100-1 # taking T to 0 to 1 limit
    
    # creating validator object
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
