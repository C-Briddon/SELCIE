#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:06:42 2021

@author: ppycb3

Environment - fenics2019

Tools to produce and modify meshes to be used in simulations.

Notes - 
    May want to compare algorithms e.g. 
    gmsh.option.setNumber('Mesh.Algorithm3D', 10) is very slow compared to default.
"""
import gmsh
import meshio
import numpy as np
import dolfin as d
from functools import partial
geom = gmsh.model.occ


class Meshing_Tools():
    def __init__(self):
        self.dim = 1
        
        self.source_boundaries = []
        self.source = []
        self.inner_wall_boundary = []
        self.outer_wall_boundary = []
        self.vacuum = []
        self.wall = []
        
        self.inner_wall = []
        self.outer_wall = []
        
        self.Min_length = 1.3e-6
        return None
    
    
    def points_to_surface(self, Points_list):
        '''
        Takes a list of points that define a closed surface and constructs
        thios surface in an open gmsh application.

        Parameters
        ----------
        Points_list : TYPE list
            List containing the points which define the exterior of the surface.
            Each element of the list should 

        Returns
        -------
        SurfaceDimTag : TYPE
            DESCRIPTION.

        '''
        
        if len(Points_list) < 3:
            print("Points_list does not have enough points to construct a surface (3 min).")
            return None
        
        Pl = []
        Ll = []
        
        'Set points.'
        for p in Points_list:
            Pl.append(geom.addPoint(p[0], p[1], p[2]))
        
        'Join points as lines.'
        for i, _ in enumerate(Points_list) :
            Ll.append(geom.addLine(Pl[i-1], Pl[i]))
        
        'Join lines as a closed loop and surface.'
        sf = geom.addCurveLoop(Ll)
        SurfaceDimTag = (2, geom.addPlaneSurface([sf]))
        
        return SurfaceDimTag
    
    
    def points_to_volume(self, Contour_list):
        
        for Points_list in Contour_list:
            if len(Points_list) < 3:
                print("One or more contours does not have enough points to",
                      "construct a surface (3 min).")
                return None
        
        
        L_list = []
        for Points_list in Contour_list:
            'Create data lists.'
            Pl = []
            Ll = []
            
            'Set points.'
            for p in Points_list:
                Pl.append(geom.addPoint(p[0], p[1], p[2]))
            
            'Join points as lines.'
            for i, _ in enumerate(Points_list):
                Ll.append(geom.addLine(Pl[i-1], Pl[i]))
            
            'Join lines as a closed loop and surface.'
            L_list.append(geom.addCurveLoop(Ll))
        
        VolumeDimTag = geom.addThruSections(L_list)
        
        
        "Delete contour lines."
        geom.remove(geom.getEntities(dim = 1), recursive = True)
        
        return VolumeDimTag
    
    
    def shape_cutoff(self, shape_DimTags, cutoff_radius = 1.0):
        '''
        Applies a radial cutoff to all shapes open in gmsh. 
        
        Parameters
        ----------
        cutoff_radius : TYPE float, optional
            The radial size of the cutoff. Any part of the source that is 
            further away from the origin than this radius will be erased. 
            The default is 1.0.
        
        Returns
        -------
        None.
        
        '''
        # Check for 3D interecting spheres.
        cutoff = [(3, geom.addSphere(xc=0, yc=0, zc=0, radius=cutoff_radius))]
        geom.intersect(objectDimTags = shape_DimTags, toolDimTags = cutoff)
        
        return None
    
    
    def construct_vacuum_chamber_2D(self, vacuum_radius = 1.0, wall_thickness = 0.1):
        '''
        Takes any 2D meshes currently in an open gmsh and places them inside a 
        circular vacuum chamber consisting of a circle of radius 'vacuum_radius'
        defining the inner wall of the chamber and a second circle for the 
        outer wall. The thickness of this wall is equal to 'wall_thickness'.
        
        Parameters
        ----------
        vacuum_radius : TYPE float, optional
            The radial size of the circle that defines the inner surface of 
            the vacuum chamber wall. The default is 1.0.
        wall_thickness : TYPE float, optional
            The width of the vacuum chamber wall. So the total domain size is 
            a circle of radius 'vacuum_radius' + 'wall_thickness'. 
            The default is 0.1.
            
        Returns
        -------
        None.
        
        '''
        
        # Get source information.
        self.source_boundaries = geom.getEntities(dim = 1)
        self.source = geom.getEntities(dim=2)
        
        
        # Define vacuum and inner wall boundary.
        vacuum_background = [(2, geom.addDisk(xc=0, yc=0, zc=0, 
                                              rx=vacuum_radius, ry=vacuum_radius))]
        
        self.vacuum, _ = geom.cut(objectDimTags = vacuum_background, 
                                  toolDimTags = self.source,
                                  removeObject = True, removeTool = False)
        
        self.inner_wall_boundary = [b for b in geom.getEntities(dim = 1)
                                    if b not in self.source_boundaries]
        
        
        # Define wall and outer wall boundary.
        wall_background = [(2, geom.addDisk(xc=0, yc=0, zc=0, 
                                            rx=vacuum_radius + wall_thickness, 
                                            ry=vacuum_radius + wall_thickness))]
        
        self.wall, _ = geom.cut(objectDimTags = wall_background, 
                                toolDimTags = self.source + self.vacuum,
                                  removeObject = True, removeTool = False)
        
        self.outer_wall_boundary = [b for b in geom.getEntities(dim = 1)
                                    if b not in self.source_boundaries + self.inner_wall_boundary]
        
        
        geom.synchronize()
        return None
    
    
    def construct_vacuum_chamber(self, dim, vacuum_radius = 1.0, wall_thickness = 0.1):
        '''
        Takes any dim-dimensional mesh currently in an open gmsh and places 
        them inside a circular/spherical vacuum chamber of radius 
        'vacuum_radius' and wall thickness 'wall_thickness'.
        
        Parameters
        ----------
        dim : TYPE int
            The dimension of the vacuum chamber.
        vacuum_radius : TYPE float, optional
            The radial size of the circle that defines the inner surface of 
            the vacuum chamber wall. The default is 1.0.
        wall_thickness : TYPE float, optional
            The width of the vacuum chamber wall. So the total domain size is 
            a circle of radius 'vacuum_radius' + 'wall_thickness'. 
            The default is 0.1.
            
        Returns
        -------
        None.
        
        '''
        
        # Get source information.
        self.source_boundaries = geom.getEntities(dim = dim-1)
        self.source = geom.getEntities(dim=dim)
        
        # Define vacuum and inner wall boundary.
        if dim == 2:
            vacuum_background = [(2, geom.addDisk(xc=0, yc=0, zc=0, 
                                                  rx=vacuum_radius, 
                                                  ry=vacuum_radius))]
        elif dim == 3:
            vacuum_background = [(3, geom.addSphere(xc=0, yc=0, zc=0, 
                                                    radius=vacuum_radius))]
        
        self.vacuum, _ = geom.cut(objectDimTags = vacuum_background, 
                                  toolDimTags = self.source,
                                  removeObject = True, removeTool = False)
        
        self.inner_wall_boundary = [b for b in geom.getEntities(dim = dim-1)
                                    if b not in self.source_boundaries]
        
        
        # Define wall and outer wall boundary.
        if dim == 2:
            wall_background = [(2, geom.addDisk(xc=0, yc=0, zc=0, 
                                                rx=vacuum_radius + wall_thickness, 
                                                ry=vacuum_radius + wall_thickness))]
        elif dim == 3:
            wall_background = [(3, geom.addSphere(xc=0, yc=0, zc=0, 
                                                  radius=vacuum_radius + wall_thickness))]
        
        self.wall, _ = geom.cut(objectDimTags = wall_background, 
                                toolDimTags = self.source + self.vacuum,
                                  removeObject = True, removeTool = False)
        
        self.outer_wall_boundary = [b for b in geom.getEntities(dim = dim-1)
                                    if b not in self.source_boundaries + self.inner_wall_boundary]
        
        
        geom.synchronize()
        return None
    
    
    def generate_mesh_2D(self, SizeMin, SizeMax, DistMin = 0.0, DistMax = 1.0, 
                         NumPointsPerCurve = 1000, refine_source_boundary = True, 
                         refine_inner_wall_boundary = True, 
                         refine_outer_wall_boundary = False):
        '''
        Generates a mesh whose cells are taged such that tag = {1, 2, 3} 
        corresponds to the source, vacuum and wall, respectively.
        
        The size of each cell is also controlled by the user such that cells 
        that are less than a distance of 'DistMin' from the source and inner
        wall boundaries will have a size of 'SizeMin', while cells with 
        distance more than 'DistMax' will have a size 'SizeMax'. For cells 
        between these two distances the cell size will increase linearly.
        The diagram below illustrates this.
        
                           DistMax
                              |
        SizeMax-             /--------
                            /
                           /
                          /
        SizeMin-    o----/
                         |
                      DistMin
        
        Parameters
        ----------
        SizeMin : TYPE float
            Minimum cell size.
        SizeMax : TYPE float
            Maximum cell size
        DistMin : TYPE, optional
            Distance from boundaries at which cell size starts to increase 
            linearly. The default is 0.0.
        DistMax : TYPE, optional
            Distance from boundaries after which cell sizes no long increase 
            linearly and are instead fixed to 'SizeMax'. The default is 1.0.
        NumPointsPerCurve : TYPE, optional
            Number of points used to define boundaries in the mesh. 
            The default is 1000.
        refine_source_boundary : TYPE, optional
            If true the source boundary will be refine as described above. 
            The default is True.
        refine_inner_wall_boundary : TYPE, optional
            If true the inner wall boundary will be refine as described above. 
            The default is True.
            
        Returns
        -------
        None.
        
        '''
        
        
        # Define distence field.
        curves_to_be_refined = []
        if refine_source_boundary:
            curves_to_be_refined += [b[1] for b in self.source_boundaries]
        
        if refine_inner_wall_boundary:
            curves_to_be_refined += [b[1] for b in self.inner_wall_boundary]
        
        if refine_outer_wall_boundary:
            curves_to_be_refined += [b[1] for b in self.outer_wall_boundary]
        
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", curves_to_be_refined)
        gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", NumPointsPerCurve)
        
        
        # Define threshold field.
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", SizeMin)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", SizeMax)
        gmsh.model.mesh.field.setNumber(2, "DistMin", DistMin)
        gmsh.model.mesh.field.setNumber(2, "DistMax", DistMax)
        
        
        # Set mesh resolution.
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)
        
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        
        
        # Mark physical domains and boundaries.
        gmsh.model.addPhysicalGroup(dim=2, tags=[s[1] for s in self.source], tag=1)
        gmsh.model.addPhysicalGroup(dim=2, tags=[s[1] for s in self.vacuum], tag=2)
        gmsh.model.addPhysicalGroup(dim=2, tags=[s[1] for s in self.wall], tag=3)
        
        gmsh.model.addPhysicalGroup(dim=1, tags=[b[1] for b in self.source_boundaries], tag=1)
        gmsh.model.addPhysicalGroup(dim=1, tags=[b[1] for b in self.inner_wall_boundary], tag=2)
        gmsh.model.addPhysicalGroup(dim=1, tags=[b[1] for b in self.outer_wall_boundary], tag=3)
        
        
        gmsh.model.mesh.generate(dim=2)
        return None
    
    
    def generate_mesh(self, dim, SizeMin, SizeMax, DistMin = 0.0, DistMax = 1.0, 
                         NumPointsPerCurve = 1000, refine_source_boundary = True, 
                         refine_inner_wall_boundary = True, 
                         refine_outer_wall_boundary = False):
        '''
        Generates a dim-dimensional mesh whose cells are taged such that 
        tag = {1, 2, 3} corresponds to the source, vacuum and wall, 
        respectively.
        
        The size of each cell is also controlled by the user such that cells 
        that are less than a distance of 'DistMin' from the source and inner
        wall boundaries will have a size of 'SizeMin', while cells with 
        distance more than 'DistMax' will have a size 'SizeMax'. For cells 
        between these two distances the cell size will increase linearly.
        The diagram below illustrates this.
        
                           DistMax
                              |
        SizeMax-             /--------
                            /
                           /
                          /
        SizeMin-    o----/
                         |
                      DistMin
        
        Parameters
        ----------
        dim : TYPE int
            Dimension of the mesh.
        SizeMin : TYPE float
            Minimum cell size.
        SizeMax : TYPE float
            Maximum cell size
        DistMin : TYPE, optional
            Distance from boundaries at which cell size starts to increase 
            linearly. The default is 0.0.
        DistMax : TYPE, optional
            Distance from boundaries after which cell sizes no long increase 
            linearly and are instead fixed to 'SizeMax'. The default is 1.0.
        NumPointsPerCurve : TYPE, optional
            Number of points used to define boundaries in the mesh. 
            The default is 1000.
        refine_source_boundary : TYPE, optional
            If true the source boundary will be refine as described above. 
            The default is True.
        refine_inner_wall_boundary : TYPE, optional
            If true the inner wall boundary will be refine as described above. 
            The default is True.
            
        Returns
        -------
        None.
        
        '''
        
        
        # Define distence field.
        curves_to_be_refined = []
        if refine_source_boundary:
            curves_to_be_refined += [b[1] for b in self.source_boundaries]
        
        if refine_inner_wall_boundary:
            curves_to_be_refined += [b[1] for b in self.inner_wall_boundary]
        
        if refine_outer_wall_boundary:
            curves_to_be_refined += [b[1] for b in self.outer_wall_boundary]
        
        if dim == 2:
            boundary_type = "CurvesList"
        elif dim == 3:
            boundary_type = "SurfacesList"
        
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, boundary_type, curves_to_be_refined)
        gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", NumPointsPerCurve)
        
        
        # Define threshold field.
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", SizeMin)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", SizeMax)
        gmsh.model.mesh.field.setNumber(2, "DistMin", DistMin)
        gmsh.model.mesh.field.setNumber(2, "DistMax", DistMax)
        
        
        # Set mesh resolution.
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)
        
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        
        
        # Mark physical domains and boundaries.
        gmsh.model.addPhysicalGroup(dim=dim, tags=[s[1] for s in self.source], tag=1)
        gmsh.model.addPhysicalGroup(dim=dim, tags=[s[1] for s in self.vacuum], tag=2)
        gmsh.model.addPhysicalGroup(dim=dim, tags=[s[1] for s in self.wall], tag=3)
        
        gmsh.model.addPhysicalGroup(dim=dim-1, tags=[b[1] for b in self.source_boundaries], tag=1)
        gmsh.model.addPhysicalGroup(dim=dim-1, tags=[b[1] for b in self.inner_wall_boundary], tag=2)
        gmsh.model.addPhysicalGroup(dim=dim-1, tags=[b[1] for b in self.outer_wall_boundary], tag=3)
        
        
        gmsh.model.mesh.generate(dim=dim)
        return None
    
    
    def msh_2_xdmf(self, filename, dim):
        '''
        Function converts .msh file (given by filename) and converts it into .xdmf
        and .h5 files which can then be used by dolfin/fenics.
        
        Returns
        mesh, subdomains
        '''
        
        # Define output filenames.
        outfile_mesh = filename + "_mesh.xdmf"
        outfile_boundary = filename + "_boundaries.xdmf"
        
        # read input from infile
        inmsh = meshio.read(filename + ".msh")
        
        if dim == 2:
            # Delete third (obj=2) column (axis=1), striping the z-component.
            outpoints = np.delete(arr=inmsh.points, obj=2, axis=1)
            
            meshio.write(outfile_mesh, 
                         meshio.Mesh(points=outpoints,
                                     cells=[('triangle', inmsh.get_cells_type('triangle'))],
                                     cell_data={'Subdomain': [inmsh.cell_data_dict['gmsh:physical']['triangle']]},
                                     field_data=inmsh.field_data))
            
            meshio.write(outfile_boundary, 
                         meshio.Mesh(points=outpoints,
                                     cells=[('line', inmsh.get_cells_type('line') )],
                                     cell_data={'Boundary': [inmsh.cell_data_dict['gmsh:physical']['line']]},
                                     field_data=inmsh.field_data))
        
        elif dim == 3:
            meshio.write(outfile_mesh, 
                         meshio.Mesh(points=inmsh.points,
                                     cells=[('tetra', inmsh.get_cells_type('tetra'))],
                                     cell_data={'Subdomain': [inmsh.cell_data_dict['gmsh:physical']['tetra']]},
                                     field_data=inmsh.field_data))
            
            meshio.write(outfile_boundary, 
                         meshio.Mesh(points=inmsh.points,
                                     cells=[('triangle', inmsh.get_cells_type('triangle') )],
                                     cell_data={'Boundary': [inmsh.cell_data_dict['gmsh:physical']['triangle']]},
                                     field_data=inmsh.field_data))
        
        
        # Import Mesh
        mesh = d.Mesh()
        with d.XDMFFile(outfile_mesh) as meshfile:
            meshfile.read(mesh)
            subdomains = d.MeshFunction('size_t', mesh, dim)
            meshfile.read(subdomains, "Subdomain")
        
        with d.XDMFFile(outfile_boundary) as boundaryfile:
            mvc = d.MeshValueCollection("size_t", mesh, dim)
            boundaryfile.read(mvc, "Boundary")
            outerwall = d.MeshFunction("size_t", mesh, mvc)
        
        return mesh, subdomains, outerwall
    
    
    def add_shapes(self, shape_1, shape_2):
        if shape_1 and shape_2:
            new_shape, _ = geom.fuse(shape_1, shape_2, removeObject = False, removeTool = False)
            
            'Get rid of unneeded shapes.'
            for shape in shape_1:
                if shape not in new_shape:
                    geom.remove([shape], recursive=True)
            
            for shape in shape_2:
                if shape not in new_shape:
                    geom.remove([shape], recursive=True)
            
        else:
            new_shape = shape_1 + shape_2
        return new_shape
    
    
    def subtract_shapes(self, shape_1, shape_2):
        if shape_1 and shape_2:
            new_shape, _ = geom.cut(shape_1, shape_2)
        else:
            new_shape = shape_1
            geom.remove(shape_2, recursive=True)
        
        return new_shape
    
    
    def intersect_shapes(self, shape_1, shape_2):
        if shape_1 and shape_2:
            new_shape, _ = geom.intersect(shape_1, shape_2)
        else:
            geom.remove(shape_1 + shape_2, recursive = True)
            new_shape = []
        return new_shape
    
    
    def non_intersect_shapes(self, shape_1, shape_2):
        "Make unit test to check this works."
        if shape_1 and shape_2:
            _, fragment_map = geom.fragment(shape_1, shape_2)
            
            shape_fragments = []
            for s in fragment_map:
                shape_fragments += s
            
            to_remove = []
            new_shape = []
            while shape_fragments:
                in_overlap = False
                for i, s in enumerate(shape_fragments[1:]):
                    if shape_fragments[0] == s:
                        to_remove.append(shape_fragments.pop(i+1))
                        in_overlap = True
                
                if in_overlap:
                    shape_fragments.pop(0)
                else:
                    new_shape.append(shape_fragments.pop(0))
            
            geom.remove(to_remove, recursive = True)
            
        else:
            geom.remove(shape_1 + shape_2, recursive = True)
            new_shape = []
        return new_shape
    
    
    def rotate_x(self, shape, rot_fraction):
        geom.rotate(shape, x=0, y=0, z=0, ax=1, ay=0, az=0, angle=2*np.pi*rot_fraction)
        return shape
    
    
    def rotate_y(self, shape, rot_fraction):
        geom.rotate(shape, x=0, y=0, z=0, ax=0, ay=1, az=0, angle=2*np.pi*rot_fraction)
        return shape
    
    
    def rotate_z(self, shape, rot_fraction):
        geom.rotate(shape, x=0, y=0, z=0, ax=0, ay=0, az=1, angle=2*np.pi*rot_fraction)
        return shape
    
    
    def translate_x(self, shape, dx):
        geom.translate(shape, dx=dx, dy=0, dz=0)
        return shape
    
    
    def translate_y(self, shape, dy):
        geom.translate(shape, dx=0, dy=dy, dz=0)
        return shape
    
    
    def translate_z(self, shape, dz):
        geom.translate(shape, dx=0, dy=0, dz=dz)
        return shape
    
    
    def unity(self, x):
        return x
    
    
    def create_disk(self, rx = 0.1, ry = 0.1):
        Rx = max(self.Min_length, abs(rx))
        Ry = max(self.Min_length, abs(ry))
        
        if Rx >= Ry:
            new_disk = [(2,geom.addDisk(xc=0, yc=0, zc=0, rx=Rx, ry=Ry))]
        else:
            new_disk = [(2,geom.addDisk(xc=0, yc=0, zc=0, rx=Ry, ry=Rx))]
            geom.rotate(new_disk, x=0, y=0, z=0, ax=0, ay=0, az=1, angle=np.pi/2)
        return new_disk
    
    
    def create_rectangle(self, dx = 0.2, dy = 0.2):
        Dx = max(self.Min_length, abs(dx))
        Dy = max(self.Min_length, abs(dy))
        
        new_rectangle = [(2,geom.addRectangle(x=-Dx/2, y=-Dy/2, z=0, dx=Dx, dy=Dy))]
        return new_rectangle
    
    
    def apply_add(self, a, b):
        return partial(self.add_shapes, a, b)()
    
    
    def apply_sub(self, a, b):
        return partial(self.subtract_shapes, a, b)()
    
    
    def apply_inx(self, a, b):
        return partial(self.intersect_shapes, a, b)()
    
    
    def apply_ninx(self, a, b):
        return partial(self.non_intersect_shapes, a, b)()
    
    
    def apply_rtx(self, a, b):
            return partial(self.rotate_x, a, b)()
        
        
    def apply_rty(self, a, b):
            return partial(self.rotate_y, a, b)()
    
    
    def apply_rtz(self, a, b):
        return partial(self.rotate_z, a, b)()
    
    
    def apply_tlx(self, a, b):
        return partial(self.translate_x, a, b)()
    
    
    def apply_tly(self, a, b):
        return partial(self.translate_y, a, b)()
    
    
    def apply_create_disk(self, a, b):
        return partial(self.create_disk, a, b)()
    
    
    def apply_create_rectangle(self, a, b):
        return partial(self.create_rectangle, a, b)()
    
    
    def apply_create_unit_disk(self):
        return partial(self.create_disk)()
    
    
    def apply_create_unit_rectangle(self):
        return partial(self.create_rectangle)()
    
    
    def shape_similarity(self, shape_1, shape_2):
        
        M1 = 0
        for s in shape_1:
            M1 += geom.getMass(dim = s[0], tag = s[1])
        
        M2 = 0
        for s in shape_2:
            M2 += geom.getMass(dim = s[0], tag = s[1])
        
        Mf = 0
        if shape_1 and shape_2:
            fragment, _ = geom.fragment(shape_1, shape_2)
            
            for f in fragment:
                Mf += geom.getMass(dim = f[0], tag = f[1])
        
        residual = abs(2*Mf - M1 - M2)
        return residual