import gmsh
import math

# --- Parameters from Experiment I Specifications (in mm) from https://arxiv.org/pdf/hep-ph/0405262 ---

# Global settings
GAP_ATTRACTOR_DISKS = 0.023
SEPARATION_S = 1.0  # Distance s (Attractor top to Pendulum bottom). 
                    # Paper range: 0.137mm to 10mm. Set to 1.0 for visualization.

# Distances below are in mm
# Vaccuum density 1e-13 g/cm^3 (residual Hydrogen)

# 1. Pendulum Ring - Aluminum density 2.77g/cm^3
P_THICKNESS = 2.002
P_HOLE_RADIUS = 4.7725
P_HOLE_DIST = 27.665
P_NUM_HOLES = 10
P_OUTER_RADIUS = 36.0 
P_INNER_RADIUS = 19.0 

# 2. Upper Attractor (UA) - High-Purity Copper 8.92 g/cm^3
UA_THICKNESS = 1.847
UA_HOLE_RADIUS = 4.7690
UA_HOLE_DIST = 27.655
UA_NUM_HOLES = 10
UA_OUTER_RADIUS = 40.0 

# 3. Lower Attractor (LA) - High-Purity Copper 8.92 g/cm^3
LA_THICKNESS = 7.828
LA_HOLE_RADIUS = 6.3449
LA_HOLE_DIST = 27.664
LA_NUM_HOLES = 10
LA_OFFSET_ANGLE = 18.0 # Degrees (Out-of-phase)
LA_OUTER_RADIUS = 40.0


def create_perforated_disk(model, z_start, thickness, outer_radius, 
                           hole_dist, hole_radius, num_holes, 
                           angle_offset_deg=0, inner_radius=0):
    """ Helper function to create a disk or ring with holes. """
    
    # 1. Create Main Disk/Ring
    if inner_radius > 0:
        cyl_outer = model.occ.addCylinder(0, 0, z_start, 0, 0, thickness, outer_radius)
        cyl_inner = model.occ.addCylinder(0, 0, z_start, 0, 0, thickness, inner_radius)
        main_body = model.occ.cut([(3, cyl_outer)], [(3, cyl_inner)])[0][0][1]
    else:
        main_body = model.occ.addCylinder(0, 0, z_start, 0, 0, thickness, outer_radius)

    # 2. Create Holes (Tool bodies)
    hole_tools = []
    offset_rad = math.radians(angle_offset_deg)
    
    for i in range(num_holes):
        angle = offset_rad + (2 * math.pi * i / num_holes)
        x = hole_dist * math.cos(angle)
        y = hole_dist * math.sin(angle)
        
        t = model.occ.addCylinder(x, y, z_start, 0, 0, thickness, hole_radius)
        hole_tools.append((3, t))

    # 3. Boolean Cut
    result_tags = model.occ.cut([(3, main_body)], hole_tools)
    return result_tags[0][0][1]

def main():
    gmsh.initialize()
    gmsh.model.add("Experiment1_NoShield")

    # --- Z-Stack Calculation ---
    # Stack: Lower Attractor -> Gap -> Upper Attractor -> Separation -> Pendulum
    z_la = 0.0
    z_ua = z_la + LA_THICKNESS + GAP_ATTRACTOR_DISKS
    z_pendulum = z_ua + UA_THICKNESS + SEPARATION_S

    # --- Build Lower Attractor ---
    print("Building Lower Attractor...")
    la_vol = create_perforated_disk(
        gmsh.model, 
        z_start=z_la, 
        thickness=LA_THICKNESS, 
        outer_radius=LA_OUTER_RADIUS, 
        hole_dist=LA_HOLE_DIST, 
        hole_radius=LA_HOLE_RADIUS, 
        num_holes=LA_NUM_HOLES,
        angle_offset_deg=LA_OFFSET_ANGLE
    )

    # --- Build Upper Attractor ---
    print("Building Upper Attractor...")
    ua_vol = create_perforated_disk(
        gmsh.model, 
        z_start=z_ua, 
        thickness=UA_THICKNESS, 
        outer_radius=UA_OUTER_RADIUS, 
        hole_dist=UA_HOLE_DIST, 
        hole_radius=UA_HOLE_RADIUS, 
        num_holes=UA_NUM_HOLES,
        angle_offset_deg=0.0
    )

    # --- Build Pendulum ---
    print("Building Pendulum...")
    p_vol = create_perforated_disk(
        gmsh.model, 
        z_start=z_pendulum, 
        thickness=P_THICKNESS, 
        outer_radius=P_OUTER_RADIUS, 
        hole_dist=P_HOLE_DIST, 
        hole_radius=P_HOLE_RADIUS, 
        num_holes=P_NUM_HOLES,
        angle_offset_deg=0.0,
        inner_radius=P_INNER_RADIUS
    )

    # --- Synchronization and Export ---
    print("Synchronizing CAD kernel...")
    gmsh.model.occ.synchronize()

    # Assign Physical Groups
    gmsh.model.addPhysicalGroup(3, [la_vol], 1, "LowerAttractor")
    gmsh.model.addPhysicalGroup(3, [ua_vol], 2, "UpperAttractor")
    gmsh.model.addPhysicalGroup(3, [p_vol], 4, "Pendulum")

    output_filename = "experiment1_simple.step"
    print(f"Writing {output_filename}...")
    gmsh.write(output_filename)
    
    print("Done. Launching GUI...")
    gmsh.fltk.run()
    
    gmsh.finalize()

if __name__ == "__main__":
    main()
