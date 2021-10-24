# SELCIE

Some modified gravity models introduce new scalar fields that couple to matter. Such fields would mediate a so called 'fifth force' which could be measured to confirm the existance of the field. So far no fifth forces have been detected. Therefore, for a scalar field to effect large scale cosmological evolution and still satisfy our constraints from local measurements (solar system), the field requires a screening mechanism. Examples include the chameleon and symmetron models. The problem with these models is that the nonlinearites introduced to produce these screening mechanisms causes the equations of motion of the field to become nonlinear and so analytic solutions only exist for highly-symmetric cases.

SELCIE (Screening Equations Linearly Constructed and Iteratively Evaluated) is a software package that provides the user with tools to investigate the chameleon model. The code provides the user with tools to construct user defined meshes by utilising the GMSH mesh generation software. These tools include constructing shapes whose boundaies are defined by some function or by constructing it out of basis shapes such as circles, cones or cylinders. The mesh can also be seperated into subdomains, each of which having its own refinement parameters. These meshes can then be converted into a format that is compatible with the finite element software FEniCS. SELCIE uses FEniCS with a nonlinear solving method (Picard or Newton method) to solve the chameleon equation of motion for some parameters and a density distrubution.


## Requirements
  - python 3.8
  - fenics 2019.1.0
  - meshio 4.3.8
  - gmsh 4.8.3
  - matplotlib
  - astropy
  - numpy
  - scipy
