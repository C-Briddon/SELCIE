#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:06:42 2021

@author: Chad Briddon

Tools to produce and modify meshes to be used in simulations.
"""
import os
import sys
import math
import gmsh
import meshio
import numpy as np
from SELCIE.Misc import legendre_R


class MeshingTools():
    def __init__(self, dimension):
        '''
        Class used to construct user-defined meshes. Creates an open gmsh
        window when class is called.

        Parameters
        ----------
        dimension : int
            The dimension of the mesh being constructed. Currently works for
            2D and 3D.

        '''

        self.dim = dimension
        self.boundaries = []
        self.subdomains = []
        self.refinement_settings = []
        self.shape_number = 0
        self.boundary_number = 0
        self.Min_length = 1.3e-6
        self.geom = gmsh.model.occ

        # Open GMSH window.
        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 1)

        return None

    def constrain_distance(self, points_list):
        '''
        Removes points from list so that the distance between neighbouring
        points is greater than the minimum gmsh line length. Is performd such
        that the last point in the list will not be removed.

        Parameters
        ----------
        points_list : list of list
            List containing the points. Each element of the list is a list
            containing the x, y, and z coordinate of the point it represents.

        Returns
        -------
        points_new : list of list
            Copy of inputted list with points removed so that all neighbouring
            points are seperated by the minimum allowed distance.

        '''

        index = []
        p_pre = points_list[-1]
        for p in points_list[:-1]:
            if math.dist(p, p_pre) < self.Min_length:
                index.append(False)
            else:
                index.append(True)
                p_pre = p

        points_new = [q for q, I in zip(points_list, index) if I]

        # Reinclude last point from original list.
        if len(points_new) > 0:
            if math.dist(points_list[-1], points_new[-1]) < self.Min_length:
                points_new[-1] = points_list[-1]
            else:
                points_new.append(points_list[-1])
        else:
            points_new.append(points_list[-1])

        return points_new

    def points_to_surface(self, points_list):
        '''
        Generates closed surface whose boundary is defined by a list of points.

        Parameters
        ----------
        points_list : list of list
            List containing the points which define the exterior of the
            surface. Each element of the list is a list containing the x, y,
            and z coordinate of the point it represents.

        Returns
        -------
        SurfaceDimTag : tuple
            Tuple containing the dimension and tag of the generated surface.

        '''

        if len(points_list) < 3:
            raise Exception(
                "'points_list' requires a minimum of 3 points.")

        Pl = []
        Ll = []

        # Set points.
        for p in points_list:
            Pl.append(self.geom.addPoint(p[0], p[1], p[2]))

        # Join points as lines.
        for i, _ in enumerate(points_list):
            Ll.append(self.geom.addLine(Pl[i-1], Pl[i]))

        # Join lines as a closed loop and surface.
        sf = self.geom.addCurveLoop(Ll)
        SurfaceDimTag = (2, self.geom.addPlaneSurface([sf]))

        return SurfaceDimTag

    def points_to_volume(self, contour_list):
        '''
        Generates closed volume whose boundary is defined by list of contours.

        Parameters
        ----------
        contour_list : list of list of list
            List containing the contours which define the exterior of the
            volume. The contours are themselves a list whose elements are
            lists, each containing the x, y, and z coordinate of the point
            it represents.

        Returns
        -------
        VolumeDimTag : tuple
            Tuple containing the dimension and tag of the generated volume.

        '''

        for points_list in contour_list:
            if len(points_list) < 3:
                raise Exception(
                    "One or more contours does not have enough points. (min 3)"
                    )

        L_list = []
        for points_list in contour_list:
            # Create data lists.
            Pl = []
            Ll = []

            # Set points.
            for p in points_list:
                Pl.append(self.geom.addPoint(p[0], p[1], p[2]))

            # Join points as lines.
            for i, _ in enumerate(points_list):
                Ll.append(self.geom.addLine(Pl[i-1], Pl[i]))

            # Join lines as a closed loop and surface.
            L_list.append(self.geom.addCurveLoop(Ll))

        VolumeDimTag = self.geom.addThruSections(L_list)

        # Delete contour lines.
        self.geom.remove(self.geom.getEntities(dim=1), recursive=True)

        return VolumeDimTag

    def construct_boundary(self, initial_boundaries, d, holes=None):
        '''
        Constructs a 2D surface around a group of closed shapes defined by
        lists of points, that is some distance 'd' away from these shapes.

        Parameters
        ----------
        initial_boundaries : list of list of list.
            Each list contains the points which define the exterior of the
            surface. Each element of the list is a list containing the x, y,
            and z coordinate of the point it represents.
        d : float
            Distance between constructed boundary and source.
        holes : list of tuple, optional
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape. The default is
            None.

        Returns
        -------
        SurfaceDimTag : list of tuple
            List of tuples repressing the new shapes. The elements of each
            tuples is the dimension and tag of the corresponding shape.

        '''

        surfaces = []
        for pos in initial_boundaries:
            boundary = []

            # Find starting point. End from max y-value.
            y = [p[1] for p in pos]
            Iy = y.index(max(y))

            # Convert list items to numpy arrays.
            pos_V = []
            for p in pos[Iy:]:
                pos_V.append(np.array(p))

            for p in pos[:Iy]:
                pos_V.append(np.array(p))

            # Add periodic point at end to prevent 'Index out of range' error.
            pos_V.append(pos_V[0])
            k_length = len(pos_V)

            # Calculate distences between consecutive points & normal vectors.
            L = [math.dist(p0, p1) for p0, p1 in zip(pos_V[:-1], pos_V[1:])]
            nv = [(p1 - p0)/l0 for p0, p1, l0 in zip(pos_V[:-1], pos_V[1:], L)]

            # Define sig and R_matrix.
            sig = 2*(np.cross(nv[-1], nv[0])[2] >= 0) - 1.0
            R_matrix = sig*d*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

            # Add periodic points into nv and L.
            L.append(L[0])
            nv.append(nv[0])

            # Add initial point.
            x_start = pos_V[0] + np.array([0.0, d, 0.0])
            start = self.geom.addPoint(x_start[0], x_start[1], x_start[2])
            I_start = start

            # Iterate over points and construct boundary.
            j = 0
            u_start = None
            while j < len(L)-1:
                # Ensure first point cannot interact with itself.
                if j == 0:
                    k_max = k_length - 1
                else:
                    k_max = k_length

                if u_start is None:
                    # Construct arc.
                    x_j, nv_j = pos_V[j], nv[j]

                    # Get tangent vector that determines if point is on chord.
                    x_end = x_j + np.matmul(R_matrix, nv_j)

                    AB = x_end - x_start
                    AC = x_j - x_start
                    T = np.cross(AB, np.cross(AB, AC))

                    # Input default solution.
                    u_k = 0.0
                    Ik = j

                    for k, (x_k, nv_k, l_k) in enumerate(zip(pos_V[j+1:k_max],
                                                             nv[j+1:],
                                                             L[j+1:])):
                        dv_jk = x_k - x_j
                        lh_jk = np.linalg.norm(dv_jk)/2

                        # Check if this arc intersects any other arcs.
                        if lh_jk <= d:
                            s = 2*(np.dot(dv_jk, nv_j) >= 0) - 1.0
                            eta = 0.5*np.sqrt(lh_jk**(-2) - d**(-2))
                            for w in [-1, +1]:
                                x_new = x_j + 0.5*dv_jk + w*s*eta*np.matmul(
                                    R_matrix, dv_jk)

                                AB = x_new - x_start
                                if np.dot(AB, T) > 0:
                                    T = np.cross(AB, np.cross(AB, AC))
                                    x_end = x_new
                                    u_k = None
                                    Ik = j + k + 1
                                    break

                        # Check if this arc intersects any lines.
                        a = np.cross(dv_jk, nv_k)[2]
                        eta = -2*sig*d*a - a**2
                        if eta >= 0:
                            s = sig*(2*(np.cross(nv_j, nv_k)[2] >= 0) - 1.0)
                            for w in [+1, -1]:
                                u_k_new = -np.dot(dv_jk, nv_k) + \
                                    w*s*np.sqrt(eta)

                                if u_k_new > 0 and u_k_new <= l_k:
                                    x_new = u_k_new*nv_k + x_k + np.matmul(
                                        R_matrix, nv_k)

                                    AB = x_new - x_start
                                    if np.dot(AB, T) > 0:
                                        T = np.cross(AB, np.cross(AB, AC))
                                        x_end = x_new
                                        u_k = u_k_new
                                        Ik = j + k + 1
                                        break

                    # If x_start and x_end are too close, then skip arc.
                    if math.dist(x_start, x_end) < 1e-7:
                        x_end = x_start
                    else:
                        # Create circle arc. Remove centre point afterwards.
                        centre = self.geom.addPoint(x_j[0], x_j[1], x_j[2])
                        end = self.geom.addPoint(x_end[0], x_end[1], x_end[2])
                        boundary.append(self.geom.addCircleArc(start, centre,
                                                               end))
                        self.geom.remove([(0, centre)])

                    j = Ik
                    u_start = u_k
                else:

                    # Construct line.
                    x_j, nv_j, l_j = pos_V[j], nv[j], L[j]

                    # Input default solution.
                    u_j = l_j
                    u_k = None
                    Ik = j + 1

                    for k, (x_k, nv_k, l_k) in enumerate(zip(pos_V[j+1:k_max],
                                                             nv[j+1:],
                                                             L[j+1:])):

                        dv_jk = x_k - x_j
                        a = np.cross(dv_jk, nv_j)[2]

                        # Check if this line intersects any circles.
                        eta = 2*sig*d*a - a**2
                        if eta >= 0:
                            u_j_new = np.dot(dv_jk, nv_j) - np.sqrt(eta)
                            if u_j_new > u_start and u_j_new < u_j:
                                u_j = u_j_new
                                u_k = None
                                Ik = j + k + 1

                        # Check if this line intersects other lines.
                        gamma = np.cross(nv_j, nv_k)[2]

                        # Check if lines are parallel.
                        if abs(gamma) > 1e-14:
                            beta = sig*d*(1 - np.dot(nv_j, nv_k))

                            u_new = (np.cross(dv_jk, nv_k)[2] + beta)/gamma

                            if u_new > u_start and u_new < u_j:
                                u_k_new = (a - beta)/gamma
                                if u_k_new > 0 and u_k_new < l_k:
                                    u_j = u_new
                                    u_k = u_k_new
                                    Ik = j + k + 1

                    # Add Line to boundary.
                    x_end = u_j*nv[j] + pos_V[j] + np.matmul(R_matrix, nv[j])
                    end = self.geom.addPoint(x_end[0], x_end[1], x_end[2])
                    boundary.append(self.geom.addLine(start, end))

                    j = Ik
                    u_start = u_k

                # Set last end point as new starting point.
                x_start = x_end
                start = end

            # Complete boundary by joining first and last points.
            centre = self.geom.addPoint(pos_V[0][0], pos_V[0][1], pos_V[0][2])
            boundary.append(self.geom.addCircleArc(start, centre, I_start))
            self.geom.remove([(0, centre)])

            # Construct mesh.
            C_b = self.geom.addCurveLoop(boundary)
            surfaces.append((2, self.geom.addPlaneSurface([C_b])))

        # Combine surfaces if more than one.
        if len(surfaces) == 1:
            SurfaceDimTag = surfaces[0:]
        else:
            SurfaceDimTag = self.geom.fuse(surfaces[:1], surfaces[1:])[0]

        # Make hole if specified.
        if holes:
            SurfaceDimTag, J1 = self.geom.cut(objectDimTags=SurfaceDimTag,
                                              toolDimTags=holes,
                                              removeObject=True,
                                              removeTool=False)

        return SurfaceDimTag

    def shape_cutoff(self, shape_DimTags, cutoff_radius=1.0):
        '''
        Applies a radial cutoff to all shapes in open gmsh window.

        Parameters
        ----------
        cutoff_radius : float, optional
            The radial size of the cutoff. Any part of a shape that is
            further away from the origin than this radius will be erased.
            The default is 1.0.

        Returns
        -------
        None.

        '''

        # Check for 3D interecting spheres.
        cutoff = [(3, self.geom.addSphere(xc=0, yc=0, zc=0,
                                          radius=cutoff_radius))]
        self.geom.intersect(objectDimTags=shape_DimTags, toolDimTags=cutoff)

        return None

    def create_subdomain(self, CellSizeMin=0.1, CellSizeMax=0.1, DistMin=0.0,
                         DistMax=1.0, NumPointsPerCurve=1000):
        '''
        Creates a subdomain from the shapes currently in an open gmsh window.
        Shapes already present in previous subdomains will not be added to the
        new one. This subdomain will be labeled by an index value corresponding
        to the next avalible integer value.

        The size of mesh cells at distances less than 'DistMin' from the
        boundary of this subdomain will be 'SizeMin', while at distances
        greater than 'DistMax' cell size is 'SizeMax'. Between 'DistMin'
        and 'DistMax' cell size will increase linearly as illustrated below.


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
        CellSizeMin : float, optional
            Minimum size of the mesh cells. The default is 0.1.
        CellSizeMax : float, optional
            Maximum size of the mesh cells. The default is 0.1.
        DistMin : float, optional
            At distances less than this value the cell size is set to its
            minimum. The default is 0.0.
        DistMax : float, optional
            At distances greater than this value the cell size is set to its
            maximum. The default is 1.0.
        NumPointsPerCurve : int, optional
            Number of points used to define each curve. The default is 1000.

        Returns
        -------
        None.

        '''

        # Save subdomains, remove duplicates, and update subdomain number.
        self.subdomains.append(
            self.geom.getEntities(dim=self.dim)[self.shape_number:])
        self.shape_number += len(self.subdomains[-1])

        # Check if new entry is empty.
        if self.subdomains[-1]:
            # Save boundary information
            self.boundaries.append(
                self.geom.getEntities(dim=self.dim-1)[self.boundary_number:])
            self.boundary_number += len(self.boundaries[-1])

            # Record refinement settings for this subdomain.
            self.refinement_settings.append([CellSizeMin, CellSizeMax, DistMin,
                                             DistMax, NumPointsPerCurve])
        else:
            del self.subdomains[-1]

        return None

    def create_background_mesh(self, CellSizeMin=0.1, CellSizeMax=0.1,
                               DistMin=0.0, DistMax=1.0,
                               NumPointsPerCurve=1000, background_radius=1.0,
                               wall_thickness=None,
                               refine_outer_wall_boundary=False):
        '''
        Generates a backgound mesh filling the space between shapes in the
        open gmsh window and a circular/spherical shell.

        The size of mesh cells at distances less than 'DistMin' from the
        boundary of this background will be 'SizeMin', while at distances
        greater than 'DistMax' cell size is 'SizeMax'. Between 'DistMin'
        and 'DistMax' cell size will increase linearly as illustrated below.


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
        CellSizeMin : float, optional
            Minimum size of the mesh cells. The default is 0.1.
        CellSizeMax : float, optional
            Maximum size of the mesh cells. The default is 0.1.
        DistMin : float, optional
            At distances less than this value the cell size is set to its
            minimum. The default is 0.0.
        DistMax : float, optional
            At distances greater than this value the cell size is set to its
            maximum. The default is 1.0.
        NumPointsPerCurve : int, optional
            Number of points used to define each curve. The default is 1000.
        background_radius : float, optional
            Radius of the circular/spherical shell used to define the
            background mesh. The default is 1.0.
        wall_thickness : None or float, optional
            If not None generates a boundary wall around the background mesh
            with specified thickness. The default is None.
        refine_outer_wall_boundary : bool, optional
            If True will also apply refinement to the exterior boundary of the
            outer wall (if exists). The default is False.

        Returns
        -------
        None.

        '''

        # Get subdomain information.
        self.create_subdomain(CellSizeMin, CellSizeMax, DistMin, DistMax,
                              NumPointsPerCurve)

        # Define vacuum and inner wall boundary.
        source_sum = self.geom.getEntities(dim=self.dim)

        if self.dim == 2:
            background_0 = [(2, self.geom.addDisk(xc=0, yc=0, zc=0,
                                                  rx=background_radius,
                                                  ry=background_radius))]
        elif self.dim == 3:
            background_0 = [(3, self.geom.addSphere(xc=0, yc=0, zc=0,
                                                    radius=background_radius))]

        if self.subdomains:
            self.geom.cut(objectDimTags=background_0, toolDimTags=source_sum,
                          removeObject=True, removeTool=False)

        # Record background as new subdomain.
        self.create_subdomain(CellSizeMin, CellSizeMax, DistMin, DistMax,
                              NumPointsPerCurve)

        # Define wall and outer wall boundary.
        if wall_thickness:
            source_sum = self.geom.getEntities(dim=self.dim)

            if self.dim == 2:
                wall_0 = [(2, self.geom.addDisk(
                    xc=0, yc=0, zc=0, rx=background_radius+wall_thickness,
                    ry=background_radius+wall_thickness))]

            elif self.dim == 3:
                wall_0 = [(3, self.geom.addSphere(
                    xc=0, yc=0, zc=0,
                    radius=background_radius+wall_thickness))]

            self.geom.cut(objectDimTags=wall_0, toolDimTags=source_sum,
                          removeObject=True, removeTool=False)

            if refine_outer_wall_boundary:
                self.create_subdomain(CellSizeMin, CellSizeMax, DistMin,
                                      DistMax, NumPointsPerCurve)
            else:
                self.create_subdomain()

        return None

    def generate_mesh(self, filename=None, show_mesh=False):
        '''
        Generate and save mesh.

        Parameters
        ----------
        filename : str, optional
            If not None then saves mesh as 'Saved Meshes'/'filename'.msh. If
            directory 'Saved Meshes' does not exist in current directory then
            one is created. The default is None.
        show_mesh : bool, optional
            If True will open a window to allow viewing of the generated mesh.
            The default is False.

        Returns
        -------
        None.

        '''

        # Catch any leftover shapes in a subdomain.
        self.create_subdomain()

        # Use fragment to align boundaries and ensure mesh isn't overlapping.
        if self.shape_number > 1:
            frag = len(self.geom.fragment(self.geom.getEntities(self.dim),
                                          [])[0])
            if frag != self.shape_number:
                print("Displaying mesh because overlap was detected.")
                print("")
                self.geom.synchronize()
                gmsh.fltk.run()
                gmsh.clear()
                gmsh.finalize()
                raise Exception("Mesh generation terminated due to overlap.")

        self.geom.synchronize()

        # If no refinement settings have been imputted then use default.
        if self.refinement_settings:

            # Get boundary_type.
            if self.dim == 2:
                boundary_type = "CurvesList"
            elif self.dim == 3:
                boundary_type = "SurfacesList"

            # Group boundaries together and define distence fields.
            i = 0
            for boundary, rf in zip(self.boundaries, self.refinement_settings):
                i += 1
                gmsh.model.mesh.field.add("Distance", i)
                gmsh.model.mesh.field.setNumbers(i, boundary_type,
                                                 [b[1] for b in boundary])
                gmsh.model.mesh.field.setNumber(i, "NumPointsPerCurve", rf[4])

            # Define threshold fields.
            j = 0
            for rf in self.refinement_settings:
                j += 1
                gmsh.model.mesh.field.add("Threshold", i+j)
                gmsh.model.mesh.field.setNumber(i+j, "InField", j)
                gmsh.model.mesh.field.setNumber(i+j, "SizeMin", rf[0])
                gmsh.model.mesh.field.setNumber(i+j, "SizeMax", rf[1])
                gmsh.model.mesh.field.setNumber(i+j, "DistMin", rf[2])
                gmsh.model.mesh.field.setNumber(i+j, "DistMax", rf[3])

            # Set mesh resolution.
            gmsh.model.mesh.field.add("Min", i+j+1)
            gmsh.model.mesh.field.setNumbers(i+j+1, "FieldsList",
                                             list(range(i+1, i+j+1)))
            gmsh.model.mesh.field.setAsBackgroundMesh(i+j+1)

            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

            # Mark physical domains and boundaries.
            for i, shape in enumerate(self.subdomains):
                gmsh.model.addPhysicalGroup(dim=self.dim,
                                            tags=[s[1] for s in shape],
                                            tag=i)

            for i, boundary in enumerate(self.boundaries):
                gmsh.model.addPhysicalGroup(dim=self.dim-1,
                                            tags=[b[1] for b in boundary],
                                            tag=i)

        # Generate mesh.
        gmsh.model.mesh.generate(dim=self.dim)

        if filename is not None:
            # If Saved Meshes directory not found create one.
            if os.path.isdir('Saved Meshes') is False:
                os.makedirs('Saved Meshes')

            gmsh.write(fileName="Saved Meshes/" + filename+".msh")

        if show_mesh is True:
            gmsh.fltk.run()

        gmsh.clear()
        gmsh.finalize()

        return None

    def msh_2_xdmf(self, filename, delete_old_file=False, auto_override=False):
        '''
        Converts .msh file into .xdmf and .h5 files which can then be used by
        dolfin/FEniCS, and saves them to a new directory with the same name as
        the .msh file.

        Parameters
        ----------
        filename : str
            Name of the .msh file that will be converted.
        delete_old_file : bool, optional
            If True will delete 'Saved Meshes'/'filename'.msh after convertion.
            The default is False.
        auto_override : bool, optional
            If True then when attempting to create the directory which contains
            the new files, if directory with this name already exists it will
            be overwritten without any user input. The default is False.

        Returns
        -------
        None.

        '''

        # Create new directory for created files.
        file_path = "Saved Meshes/" + filename
        try:
            os.makedirs(file_path)
        except FileExistsError:
            if auto_override is False:
                print('Directory %s already exists.' % file_path)
                ans = input('Overwrite files inside this directory? (y/n) ')

                if ans.lower() == 'y':
                    pass
                elif ans.lower() == 'n':
                    sys.exit()
                else:
                    print('Invalid input.')
                    sys.exit()

        # Define output filenames.
        outfile_mesh = file_path + "/mesh.xdmf"
        outfile_boundary = file_path + "/boundaries.xdmf"

        # read input from infile
        inmsh = meshio.read(file_path + ".msh")

        if self.dim == 2:
            # Delete third (obj=2) column (axis=1), striping the z-component.
            outpoints = np.delete(arr=inmsh.points, obj=2, axis=1)

            meshio.write(
                outfile_mesh, meshio.Mesh(
                    points=outpoints,
                    cells=[('triangle', inmsh.get_cells_type('triangle'))],
                    cell_data={'Subdomain': [inmsh.cell_data_dict[
                        'gmsh:physical']['triangle']]},
                    field_data=inmsh.field_data))

            meshio.write(
                outfile_boundary, meshio.Mesh(
                    points=outpoints,
                    cells=[('line', inmsh.get_cells_type('line'))],
                    cell_data={'Boundary': [inmsh.cell_data_dict[
                        'gmsh:physical']['line']]},
                    field_data=inmsh.field_data))

        elif self.dim == 3:
            meshio.write(
                outfile_mesh, meshio.Mesh(
                    points=inmsh.points,
                    cells=[('tetra', inmsh.get_cells_type('tetra'))],
                    cell_data={'Subdomain': [inmsh.cell_data_dict[
                        'gmsh:physical']['tetra']]},
                    field_data=inmsh.field_data))

            meshio.write(
                outfile_boundary, meshio.Mesh(
                    points=inmsh.points,
                    cells=[('triangle', inmsh.get_cells_type('triangle'))],
                    cell_data={'Boundary': [inmsh.cell_data_dict[
                        'gmsh:physical']['triangle']]},
                    field_data=inmsh.field_data))

        # Delete .msh file.
        if delete_old_file is True:
            os.remove(file_path + ".msh")

        return None

    def add_shapes(self, shapes_1, shapes_2):
        '''
        Fusses together elements of 'shapes_1' and 'shapes_2' to form new group
        of shapes.

        Parameters
        ----------
        shapes_1, shapes_2 : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.

        Returns
        -------
        new_shapes : list of tuple
            List of tuples repressing the new group of shapes.

        '''

        if shapes_1 and shapes_2:
            new_shapes, _ = self.geom.fuse(shapes_1, shapes_2,
                                           removeObject=False,
                                           removeTool=False)

            # Get rid of unneeded shapes.
            for shape in shapes_1:
                if shape not in new_shapes:
                    self.geom.remove([shape], recursive=True)

            for shape in shapes_2:
                if shape not in new_shapes:
                    self.geom.remove([shape], recursive=True)

        else:
            new_shapes = shapes_1 + shapes_2

        return new_shapes

    def subtract_shapes(self, shapes_1, shapes_2):
        '''
        Subtracts elements of 'shapes_2' from 'shapes_1' to form new group of
        shapes.

        Parameters
        ----------
        shapes_1, shapes_2 : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.

        Returns
        -------
        new_shapes : list of tuple
            List of tuples repressing the new group of shapes.

        '''

        if shapes_1 and shapes_2:
            new_shapes, _ = self.geom.cut(shapes_1, shapes_2)
        else:
            new_shapes = shapes_1
            self.geom.remove(shapes_2, recursive=True)

        return new_shapes

    def intersect_shapes(self, shapes_1, shapes_2):
        '''
        Creates group of shapes consisting of the intersection of elements
        from 'shapes_1' and 'shapes_2'.

        Parameters
        ----------
        shapes_1, shapes_2 : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.

        Returns
        -------
        new_shapes : list of tuple
            List of tuples repressing the new group of shapes.

        '''

        if shapes_1 and shapes_2:
            new_shapes, _ = self.geom.intersect(shapes_1, shapes_2)
        else:
            self.geom.remove(shapes_1 + shapes_2, recursive=True)
            new_shapes = []

        return new_shapes

    def non_intersect_shapes(self, shapes_1, shapes_2):
        '''
        Creates group of shapes consisting of the non-intersection of elements
        from 'shapes_1' and 'shapes_2'.

        Parameters
        ----------
        shapes_1, shapes_2 : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.

        Returns
        -------
        new_shapes : list of tuple
            List of tuples repressing the new group of shapes.

        '''

        if shapes_1 and shapes_2:
            _, fragment_map = self.geom.fragment(shapes_1, shapes_2)

            shape_fragments = []
            for s in fragment_map:
                shape_fragments += s

            to_remove = []
            new_shapes = []
            while shape_fragments:
                in_overlap = False
                for i, s in enumerate(shape_fragments[1:]):
                    if shape_fragments[0] == s:
                        to_remove.append(shape_fragments.pop(i+1))
                        in_overlap = True

                if in_overlap:
                    shape_fragments.pop(0)
                else:
                    new_shapes.append(shape_fragments.pop(0))

            self.geom.remove(to_remove, recursive=True)

        else:
            self.geom.remove(shapes_1 + shapes_2, recursive=True)
            new_shapes = []

        return new_shapes

    def rotate_x(self, shapes, rot_fraction):
        '''
        Rotates group of shapes around the x-axis.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        rot_fraction : float
            Fraction of a full rotation the group will be rotated by.

        Returns
        -------
        shapes : list tuple
            List of tuples repressing the group of shapes. Is identical to
            input 'shapes'.

        '''

        self.geom.rotate(shapes, x=0, y=0, z=0, ax=1, ay=0, az=0,
                         angle=2*np.pi*rot_fraction)

        return shapes

    def rotate_y(self, shapes, rot_fraction):
        '''
        Rotates group of shapes around the y-axis.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        rot_fraction : float
            Fraction of a full rotation the group will be rotated by.

        Returns
        -------
        shapes : list tuple
            List of tuples repressing the group of shapes. Is identical to
            input 'shapes'.

        '''

        self.geom.rotate(shapes, x=0, y=0, z=0, ax=0, ay=1, az=0,
                         angle=2*np.pi*rot_fraction)

        return shapes

    def rotate_z(self, shapes, rot_fraction):
        '''
        Rotates group of shapes around the z-axis.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        rot_fraction : float
            Fraction of a full rotation the group will be rotated by.

        Returns
        -------
        shapes : list tuple
            List of tuples repressing the group of shapes. Is identical to
            input 'shapes'.

        '''

        self.geom.rotate(shapes, x=0, y=0, z=0, ax=0, ay=0, az=1,
                         angle=2*np.pi*rot_fraction)

        return shapes

    def translate_x(self, shapes, dx):
        '''
        Translates group of shapes in the x-direction.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        dx : float
            Amount the group of shapes is to be translated by in the posative
            x-direction. If negative then translation will be in the negative
            x-direction.

        Returns
        -------
        shapes : list tuple
            List of tuples repressing the group of shapes. Is identical to
            input 'shapes'.

        '''

        self.geom.translate(shapes, dx=dx, dy=0, dz=0)

        return shapes

    def translate_y(self, shapes, dy):
        '''
        Translates group of shapes in the y-direction.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        dy : float
            Amount the group of shapes is to be translated by in the posative
            y-direction. If negative then translation will be in the negative
            y-direction.

        Returns
        -------
        shapes : list tuple
            List of tuples repressing the group of shapes. Is identical to
            input 'shapes'.

        '''

        self.geom.translate(shapes, dx=0, dy=dy, dz=0)

        return shapes

    def translate_z(self, shapes, dz):
        '''
        Translates group of shapes in the z-direction.

        Parameters
        ----------
        shapes : list of tuple
            List of tuples repressing a groups of shapes. Each tuple contains
            the dimension and tag of its corresponding shape.
        dz : float
            Amount the group of shapes is to be translated by in the posative
            z-direction. If negative then translation will be in the negative
            z-direction.

        Returns
        -------
        shapes : list tuple
            List of tuples repressing the group of shapes. Is identical to
            input 'shapes'.

        '''

        self.geom.translate(shapes, dx=0, dy=0, dz=dz)

        return shapes

    def create_ellipse(self, rx=0.1, ry=0.1):
        '''
        Generates an ellipse in an open gmsh window with its centre of mass at
        the origin.

        Parameters
        ----------
        rx : float, optional
            Ellipse radial size along x-axis. The default is 0.1.
        ry : float, optional
            Ellipse radial size along y-axis. The default is 0.1.

        Returns
        -------
        ellipse : list tuple
            List containing tuple representing the ellipse.

        '''

        Rx = max(self.Min_length, abs(rx))
        Ry = max(self.Min_length, abs(ry))

        if Rx >= Ry:
            ellipse = [(2, self.geom.addDisk(xc=0, yc=0, zc=0, rx=Rx, ry=Ry))]
        else:
            ellipse = [(2, self.geom.addDisk(xc=0, yc=0, zc=0, rx=Ry, ry=Rx))]
            self.geom.rotate(ellipse, x=0, y=0, z=0, ax=0, ay=0, az=1,
                             angle=np.pi/2)

        return ellipse

    def create_rectangle(self, dx=0.2, dy=0.2):
        '''
        Generates a rectangle in an open gmsh window with its centre of mass at
        the origin.

        Parameters
        ----------
        dx : float, optional
            Length of rectangle along x-axis. The default is 0.2.
        dy : float, optional
            Length of rectangle along y-axis. The default is 0.2.

        Returns
        -------
        rectangle : list tuple
            List containing tuple representing the rectangle.

        '''

        Dx = max(self.Min_length, abs(dx))
        Dy = max(self.Min_length, abs(dy))

        rectangle = [(2, self.geom.addRectangle(x=-Dx/2, y=-Dy/2, z=0,
                                                dx=Dx, dy=Dy))]

        return rectangle

    def create_ellipsoid(self, rx=0.1, ry=0.1, rz=0.1):
        '''
        Generates an ellipsoid in an open gmsh window with its centre of mass
        at the origin.

        Parameters
        ----------
        rx : float, optional
            Ellipsoid radial size along x-axis. The default is 0.1.
        ry : float, optional
            Ellipsoid radial size along y-axis. The default is 0.1.
        rz : float, optional
            Ellipsoid radial size along z-axis. The default is 0.1.

        Returns
        -------
        ellipsoid : list tuple
            List containing tuple representing the ellipsoid.

        '''

        Rx = max(self.Min_length, abs(rx))
        Ry = max(self.Min_length, abs(ry))
        Rz = max(self.Min_length, abs(rz))

        ellipsoid = [(3, self.geom.addSphere(xc=0, yc=0, zc=0, radius=1))]
        self.geom.dilate(ellipsoid, x=0, y=0, z=0, a=Rx, b=Ry, c=Rz)

        return ellipsoid

    def create_box(self, dx=0.2, dy=0.2, dz=0.2):
        '''
        Generates a box in an open gmsh window with its centre of mass at the
        origin.

        Parameters
        ----------
        dx : float, optional
            Length of box along x-axis. The default is 0.2.
        dy : float, optional
            Length of box along y-axis. The default is 0.2.
        dz : float, optional
            Length of box along z-axis. The default is 0.2.

        Returns
        -------
        box : list tuple
            List containing tuple representing the box.

        '''

        Dx = max(self.Min_length, abs(dx))
        Dy = max(self.Min_length, abs(dy))
        Dz = max(self.Min_length, abs(dz))

        box = [(3, self.geom.addBox(x=-Dx/2, y=-Dy/2, z=-Dz/2, dx=Dx,
                                    dy=Dy, dz=Dz))]

        return box

    def create_cylinder(self, Length=0.1, r=0.1):
        '''
        Generates a cylinder in an open gmsh window with its centre of mass at
        the origin.

        Parameters
        ----------
        Length : float, optional
            Length of cylinder. The default is 0.1.
        r : float, optional
            Radial size of cylinder. The default is 0.1.

        Returns
        -------
        cylinder : list tuple
            List containing tuple representing the cylinder.

        '''

        L = max(self.Min_length, abs(Length))
        R = max(self.Min_length, abs(r))

        cylinder = [(3, self.geom.addCylinder(x=0, y=0, z=-L/2, dx=0, dy=0,
                                              dz=L, r=R))]

        return cylinder

    def create_cone(self, Length=0.1, r=0.1):
        '''
        Generates a cone in an open gmsh window with its centre of mass at
        the origin.

        Parameters
        ----------
        Length : float, optional
            Length between tip and base of the cone. The default is 0.1.
        r : float, optional
            Radial size at the base of the cone. The default is 0.1.

        Returns
        -------
        cone : list tuple
            List containing tuple representing the cone.

        '''

        L = max(self.Min_length, abs(Length))
        R = max(self.Min_length, abs(r))

        cone = [(3, self.geom.addCone(x=0, y=0, z=-L/4, dx=0, dy=0, dz=L,
                                      r1=R, r2=0))]

        return cone

    def create_torus(self, r_hole=0.1, r_tube=0.1):
        '''
        Generates a torus in an open gmsh window with its centre of mass at
        the origin.

        Parameters
        ----------
        r_hole : float, optional
            Radius of hole through centre of the torus. The default is 0.1.
        r_tube : float, optional
            Radius of the torus tube. The default is 0.1.

        Returns
        -------
        torus : list tuple
            List containing tuple representing the torus.

        '''

        R_hole = max(self.Min_length, abs(r_hole))
        R_tube = max(self.Min_length, abs(r_tube))

        torus = [(3, self.geom.addTorus(x=0, y=0, z=0, r1=R_hole+R_tube,
                                        r2=R_tube))]

        return torus

    def legendre_shape_components(self, a_coef, angle=2*np.pi, N=100):
        '''
        Generate lists of points that outline segments of the Legendre
        polynomial shape.

        Parameters
        ----------
        a_coef : list of float
            Coefficients of the Legendre series.
        angle : float, optional
            Angular amount the shape covers in radians. The default is 2*np.pi.
        N : int, optional
            Total number of points. The default is 100.

        Returns
        -------
        shapes_pos : list of list of list
            Each list in shapes_pos contains lists representing a point with a
            positive radial value from the Legendre series. The elements of the
            lists are the x, y, and z coordinates of the point.
        shapes_neg : list of list of list
            Each list in shapes_neg contains lists representing a point with a
            negative radial value from the Legendre series. The elements of the
            lists are the x, y, and z coordinates of the point.

        '''

        theta = np.linspace(0, angle, N, endpoint=False)
        R = legendre_R(theta, a_coef)
        dR = np.diff(R, append=R[0])

        # Group x, y, and z components of each point and split curve.
        split = (R*(R + dR) < 0.0)

        curves = np.array(list(zip(R*np.sin(theta),
                                   R*np.cos(theta),
                                   np.zeros(len(R)))))

        shapes_holes = [ar.tolist() for ar in
                        np.array_split(curves, np.where(split)[0]+1)]

        # Join first and last lines to complete segment.
        if len(shapes_holes) > 1:
            for sh in shapes_holes[:-1]:
                sh.append([0.0, 0.0, 0.0])
            shapes_holes[0] += shapes_holes.pop(-1)

        # Apply distance constraint to each shape.
        for i, sh in enumerate(shapes_holes):
            shapes_holes[i] = self.constrain_distance(sh)

        # Create lists for segments with positive and negative R.
        shapes_pos = [sh for sh in shapes_holes[0::2] if len(sh) > 2]
        shapes_neg = [sh for sh in shapes_holes[1::2] if len(sh) > 2]

        return shapes_pos, shapes_neg

    def construct_legendre_mesh_2D(self, a_coef, angle=2*np.pi, N=100,
                                   include_holes=True):
        '''
        Generate a 2D shape constructed using a Legendre series.

        Parameters
        ----------
        a_coef : list of float
            Coefficients of the Legendre series.
        angle : float, optional
            Angular amount the shape covers in radians. The default is 2*np.pi.
        N : int, optional
            Total number of points. The default is 100.
        include_holes : bool, optional
            If True will remove intersections of regions with posative and
            negative radius from shape. Otherwise the shapes are merged so
            they have no holes. The default is True.

        Returns
        -------
        SurfaceDimTag : list of tuple
            List containing Tuples which contain the dimension and tag of the
            generated surface.

        '''

        shapes_pos, shapes_neg = self.legendre_shape_components(a_coef,
                                                                angle, N)

        PosDimTags = []
        for shape in shapes_pos:
            PosDimTags.append(self.points_to_surface(points_list=shape))

        if include_holes:
            NegDimTags = []
            for shape in shapes_neg:
                NegDimTags.append(self.points_to_surface(points_list=shape))

            if PosDimTags and NegDimTags:
                SurfaceDimTags = self.non_intersect_shapes(PosDimTags,
                                                           NegDimTags)
            else:
                SurfaceDimTags = PosDimTags + NegDimTags
        else:
            SurfaceDimTags = PosDimTags

        return SurfaceDimTags
