import gmsh
import math

# --- Parameters from Experiment II Specifications (Table II in paper) from https://arxiv.org/pdf/hep-ph/0405262 ---

# Global Z-stacking
GAP_ATTRACTOR_DISKS = 0.0008 # Fitted value from paper
SEPARATION_S = 1.0           # Arbitrary visualization distance (s)

# Distances below are in mm
# Vaccuum density 1e-13 g/cm^3 (residual Hydrogen)

# 1. Pendulum Ring - Aluminum density 2.77g/cm^3
P_THICKNESS = 2.979
P_HOLE_RADIUS = 3.1873
P_HOLE_DIST = 26.667
P_NUM_HOLES = 10
# Dimensions for the ring body (inferred to fit holes)
P_OUTER_RADIUS = 35.0 
P_INNER_RADIUS = 18.0 

# 2. Upper Attractor (UA) - This has TWO sets of holes - High-Purity Copper 8.92 g/cm^3
UA_THICKNESS = 3.005
UA_OUTER_RADIUS = 50.0 # Needs to be larger to accommodate outer holes (r=38.7 + a=4.8)

# UA Set 1: In-Phase (Aligned with Pendulum)
UA_IN_RADIUS = 3.1930
UA_IN_DIST = 26.685
UA_IN_ANGLE = 0.0

# UA Set 2: Out-of-Phase
UA_OUT_RADIUS = 4.7720
UA_OUT_DIST = 38.697
UA_OUT_ANGLE = 18.0 # Degrees (180/10)

# 3. Lower Attractor (LA) - High-Purity Copper 8.92 g/cm^3
LA_THICKNESS = 6.998
LA_OUTER_RADIUS = 50.0
LA_HOLE_RADIUS = 7.9693
LA_HOLE_DIST = 36.666
LA_ANGLE = 18.0 # Out-of-Phase

def create_ring_or_disk(model, z_start, thickness, outer_radius, inner_radius=0):
    """ Creates the base solid body (Cylinder or Ring) """
    cyl_outer = model.occ.addCylinder(0, 0, z_start, 0, 0, thickness, outer_radius)
    
    if inner_radius > 0:
        cyl_inner = model.occ.addCylinder(0, 0, z_start, 0, 0, thickness, inner_radius)
        # Cut inner from outer to make ring
        result = model.occ.cut([(3, cyl_outer)], [(3, cyl_inner)])
        return result[0][0][1] # Return tag of the resulting ring
    else:
        return cyl_outer

def cut_hole_pattern(model, body_tag, z_start, thickness, 
                     hole_dist, hole_radius, num_holes, angle_offset_deg):
    """ 
    Cuts a pattern of holes into an existing body.
    Returns the tag of the modified body.
    """
    hole_tools = []
    offset_rad = math.radians(angle_offset_deg)
    
    for i in range(num_holes):
        # Calculate center of hole
        angle = offset_rad + (2 * math.pi * i / num_holes)
        x = hole_dist * math.cos(angle)
        y = hole_dist * math.sin(angle)
        
        # Create cylinder for the hole
        t = model.occ.addCylinder(x, y, z_start, 0, 0, thickness, hole_radius)
        hole_tools.append((3, t))

    # Perform Boolean Cut (Body - Holes)
    # Note: We wrap body_tag in list of tuples [(3, tag)]
    result = model.occ.cut([(3, body_tag)], hole_tools)
    
    # Return the new tag of the cut object
    return result[0][0][1]

def main():
    gmsh.initialize()
    gmsh.model.add("Experiment2")

    # --- Z-Stack Calculation ---
    # Stack: Lower Attractor -> Gap -> Upper Attractor -> Separation -> Pendulum
    z_la = 0.0
    z_ua = z_la + LA_THICKNESS + GAP_ATTRACTOR_DISKS
    z_pendulum = z_ua + UA_THICKNESS + SEPARATION_S

    # --- Build Lower Attractor ---
    print("Building Lower Attractor...")
    # 1. Create solid disk
    la_body = create_ring_or_disk(gmsh.model, z_la, LA_THICKNESS, LA_OUTER_RADIUS)
    # 2. Cut holes (Out-of-phase)
    la_final = cut_hole_pattern(gmsh.model, la_body, z_la, LA_THICKNESS, 
                                LA_HOLE_DIST, LA_HOLE_RADIUS, P_NUM_HOLES, LA_ANGLE)

    # --- Build Upper Attractor ---
    print("Building Upper Attractor (Two hole sets)...")
    # 1. Create solid disk
    ua_body = create_ring_or_disk(gmsh.model, z_ua, UA_THICKNESS, UA_OUTER_RADIUS)
    # 2. Cut Set 1: In-Phase Holes
    ua_cut_1 = cut_hole_pattern(gmsh.model, ua_body, z_ua, UA_THICKNESS,
                                UA_IN_DIST, UA_IN_RADIUS, P_NUM_HOLES, UA_IN_ANGLE)
    # 3. Cut Set 2: Out-of-Phase Holes (using the resulting body from previous cut)
    ua_final = cut_hole_pattern(gmsh.model, ua_cut_1, z_ua, UA_THICKNESS,
                                UA_OUT_DIST, UA_OUT_RADIUS, P_NUM_HOLES, UA_OUT_ANGLE)

    # --- Build Pendulum ---
    print("Building Pendulum...")
    # 1. Create ring
    p_body = create_ring_or_disk(gmsh.model, z_pendulum, P_THICKNESS, 
                                 P_OUTER_RADIUS, inner_radius=P_INNER_RADIUS)
    # 2. Cut holes (In-Phase)
    p_final = cut_hole_pattern(gmsh.model, p_body, z_pendulum, P_THICKNESS,
                               P_HOLE_DIST, P_HOLE_RADIUS, P_NUM_HOLES, 0.0)

    # --- Synchronization and Export ---
    print("Synchronizing CAD kernel...")
    gmsh.model.occ.synchronize()

    # Assign Physical Groups
    gmsh.model.addPhysicalGroup(3, [la_final], 1, "LowerAttractor")
    gmsh.model.addPhysicalGroup(3, [ua_final], 2, "UpperAttractor")
    gmsh.model.addPhysicalGroup(3, [p_final], 4, "Pendulum")

    output_filename = "experiment2_simple.step"
    print(f"Writing {output_filename}...")
    gmsh.write(output_filename)
    
    print("Done. Launching GUI...")
    gmsh.fltk.run()
    
    gmsh.finalize()

if __name__ == "__main__":
    main()
