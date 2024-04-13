import argparse
import math

def gen_blockMeshDict(a, b):
    """
    Create a `blockMeshDict` file for the geometry
    """

    scale = 1
    z = 0.05
    y_max = 1
    a = a
    b = b
    x_mesh = a + 0.2
    x_min = -x_mesh - 4
    x_max = x_min + 15 + a

    x_cell_pre = int( (abs(x_min) - x_mesh)* 50/2)
    y_cell_post_pre = int( 2 * y_max * 50)
    x_cell_post = int( ( x_max - x_mesh ) * 20)
    x_cell_in = int( 2* x_mesh * 50)
    y_cell_in = y_cell_post_pre
    x_cell_common = int((x_mesh) * 50)
    

    def get_x(a,b,theta):
     x = a*b/(math.sqrt( (b*math.cos(theta))**2 + (a*math.sin(theta))**2 ) ) * math.cos(theta)
     return x

    def get_y(a,b,theta):
     y= a*b/(math.sqrt( (b*math.cos(theta))**2 + (a*math.sin(theta))**2 ) ) * math.sin(theta)
     return y

    x_y_45 = get_x(a,b,math.pi/4)
    x_5 = get_x(a,b,math.pi*5/180)
    y_5 = get_y(a,b,math.pi*5/180)
    x_10 = get_x(a,b,math.pi*10/180)
    y_10 = get_y(a,b,math.pi*10/180)
    x_20 = get_x(a,b,math.pi*20/180)
    y_20 = get_y(a,b,math.pi*20/180)
    x_30 = get_x(a,b,math.pi*30/180)
    y_30 = get_y(a,b,math.pi*30/180)
    x_40 = get_x(a,b,math.pi*40/180)
    y_40 = get_y(a,b,math.pi*40/180)

    # Open file
    f = open("blockMeshDict", "w")

    # Write file
    f.write("/*--------------------------------*- C++ -*----------------------------------*\ \n"
            "| =========                |                                                  |\n"
            "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox            |\n"
            "|  \\    /   O peration     | Version:  5                                      |\n"
            "|   \\  /    A nd           | Web:      www.OpenFOAM.org                       |\n"
            "|    \\/     M anipulation  |                                                  |\n"
            "\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n"
            "{\n"
            "   version     2.0;\n"
            "   format      ascii;\n"
            "   class       dictionary;\n"
            "   object      blockMeshDict;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n")
    f.write("convertToMeters {};\n".format(scale))
    f.write("\n"
            "vertices\n"
            "("
            "\n")
    f.write("    ({} {} {})\n".format(x_min, -y_max, z)) #0
    f.write("    ({} {} {})\n".format(-x_mesh, -y_max, z)) #1
    f.write("    ({} {} {})\n".format(-x_mesh, -y_max, -z)) #2
    f.write("    ({} {} {})\n".format(x_min, -y_max, -z)) #3

    f.write("    ({} {} {})\n".format(x_min, y_max, z)) #4 
    f.write("    ({} {} {})\n".format(-x_mesh , y_max, z)) #5
    f.write("    ({} {} {})\n".format(-x_mesh, y_max, -z)) #6 
    f.write("    ({} {} {})\n".format(x_min, y_max, -z)) #7

    f.write("    ({} {} {})\n".format(x_mesh, -y_max, z)) #8
    f.write("    ({} {} {})\n".format(x_mesh, -y_max, -z)) #9
  
    f.write("    ({} {} {})\n".format(-x_y_45, -x_y_45, z)) #10
    f.write("    ({} {} {})\n".format(x_y_45, -x_y_45, z)) #11
    f.write("    ({} {} {})\n".format(x_y_45, -x_y_45, -z)) #12
    f.write("    ({} {} {})\n".format(-x_y_45, -x_y_45, -z)) #13
    f.write("    ({} {} {})\n".format(-x_y_45, x_y_45, z)) #14
    f.write("    ({} {} {})\n".format(-x_y_45, x_y_45, -z)) #15
    f.write("    ({} {} {})\n".format(x_y_45, x_y_45, z)) #16
    f.write("    ({} {} {})\n".format(x_y_45, x_y_45, -z)) #17


    f.write("    ({} {} {})\n".format(x_mesh, y_max, z)) #18
    f.write("    ({} {} {})\n".format(x_mesh, y_max, -z)) #19

    f.write("    ({} {} {})\n".format(x_max, -y_max, z)) #20
    f.write("    ({} {} {})\n".format(x_max, -y_max, -z)) #21
    f.write("    ({} {} {})\n".format(x_max, y_max, z)) #22
    f.write("    ({} {} {})\n".format(x_max, y_max, -z)) #23 

    f.write(");\n"
            "\n"
            "blocks\n"
            "(\n")

    # pre-block
    f.write("    hex ( 0  1  2  3  4  5  6  7) ({} {} {}) simpleGrading (1 10 1)\n".format(x_cell_pre, 1 ,y_cell_post_pre))
    # obstacle blocks
    f.write("    hex ( 1  8  9  2 10 11 12 13) ({} {} {}) simpleGrading (1 1 0.1)\n".format(x_cell_in, 1, x_cell_common))
    f.write("    hex ( 1 10 13  2  5 14 15  6) ({} {} {}) simpleGrading (0.1 1 1)\n".format(x_cell_common, 1 , y_cell_post_pre))
    f.write("    hex (14 16 17 15  5 18 19  6) ({} {} {}) simpleGrading (1 1 10)\n".format(x_cell_in , 1 , x_cell_common))
    f.write("    hex (11  8  9 12 16 18 19 17) ({} {} {}) simpleGrading (10 1 1)\n".format(x_cell_common, 1,  y_cell_post_pre))

    # post-block
    f.write("    hex ( 8 20 21  9 18 22 23 19) ({} {} {}) simpleGrading (5 10 1)\n".format(x_cell_post, 1 , y_cell_post_pre))



    f.write(");\n")
    f.write("\n"
            "edges\n"
            "(\n")
    f.write("    spline 14 10 (({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}))\n".format(-x_40,y_40,z,-x_30,y_30,z,-x_20,y_20,z,-x_10,y_10,z,-x_5,y_5,z,-a,0,z,-x_5,-y_5,z,-x_10,-y_10,z,-x_20,-y_20,z,-x_30,-y_30,z,-x_40,-y_40,z)) 
    f.write("    spline 15 13 (({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}))\n".format(-x_40,y_40,-z,-x_30,y_30,-z,-x_20,y_20,-z,-x_10,y_10,-z,-x_5,y_5,-z,-a,0,-z,-x_5,-y_5,-z,-x_10,-y_10,-z,-x_20,-y_20,-z,-x_30,-y_30,-z,-x_40,-y_40,-z))  
    f.write("    spline 16 11 (({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}))\n".format(x_40,y_40,z,x_30,y_30,z,x_20,y_20,z,x_10,y_10,z,x_5,y_5,z,a,0,z,x_5,-y_5,z,x_10,-y_10,z,x_20,-y_20,z,x_30,-y_30,z,x_40,-y_40,z))  
    f.write("    spline 17 12 (({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}) ({} {} {}))\n".format(x_40,y_40,-z,x_30,y_30,-z,x_20,y_20,-z,x_10,y_10,-z,x_5,y_5,-z,a,0,-z,x_5,-y_5,-z,x_10,-y_10,-z,x_20,-y_20,-z,x_30,-y_30,-z,x_40,-y_40,-z))   


    f.write("    arc 10 11 ({} {} {})\n".format(0, -b, z))
    f.write("    arc 12 13 ({} {} {})\n".format(0, -b, -z))
    f.write("    arc 14 16 ({} {} {})\n".format(0, b, z))
    f.write("    arc 15 17 ({} {} {})\n".format(0, b, -z))


    f.write(");\n"
            "\n")
    f.write("boundary\n"
            "(\n"
            "    inlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            ( 0  4  7  3)\n"
            "        );\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "		 (20 21 23 22)\n"
            "        );\n"
            "    }\n"
            "    top\n"
            "    {\n"
            "        type wall;\n"
            "        faces\n"
            "        (\n"
            "            ( 4  5  6  7)\n"
            "            ( 5 18 19  6)\n"
            "            (18 22 23 19)\n"
            "            ( 0  1  2  3)\n"
            "            ( 1  8  9  2)\n"
            "            ( 8 20 21  9)\n"
            "        );\n"
            "    }\n"
            "    obstacle\n"
            "    {\n"
            "        type wall;\n"
            "        faces\n"
            "        (\n"
            "            (10 11 12 13)\n"
            "            (11 16 17 12)\n"
            "            (14 16 17 15)\n"
            "            (10 14 15 13)\n"
            "        );\n"
            "    }\n"
            ");\n"
            "\n"
            "mergePatchPairs\n"
            "(\n"
            ");\n"
            "\n"
            "// ************************************************************************* //\n")

    # Close file
    f.close()



# Total cell 7500

parser = argparse.ArgumentParser(description="Generating blockMeshDict file for the geometry")
parser.add_argument("a", help="x_axis dim")
parser.add_argument("b", help="y_axis dim")
args = parser.parse_args()
gen_blockMeshDict( float(args.a), float(args.b) )

