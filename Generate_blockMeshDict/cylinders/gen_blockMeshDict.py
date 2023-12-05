 import argparse


def gen_blockMeshDict(r_int, y_max):
    """
    Create a `blockMeshDict` file for the geometry
    """

    scale_for_cells = 40

    scale = 1
    z = 0.05
    x_orig = 0
    y_orig = 0
    ymax = y_max
    r_int = r_int
    xmin = -r_int - 4.0
    xmax = xmin + 15
    r_int = r_int
    r_ext = 2 * r_int
    x_cell = int(r_int * scale_for_cells*4+5)
    x_cell2 = int( (xmax - r_ext) * scale_for_cells*2 /2)
    x_cell3 = int( ( abs(xmin) - r_ext ) * scale_for_cells*2 /2)
    y_cell = int(r_int * scale_for_cells*4+5)
    y_cell2 = int( (ymax - r_ext) * scale_for_cells*4 + 5)

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
    f.write("    ({} {} {})\n".format(r_int, y_orig, -z)) #0
    f.write("    ({} {} {})\n".format(r_ext, y_orig, -z)) #1
    f.write("    ({} {} {})\n".format(xmax, y_orig, -z)) #2
    f.write("    ({} {} {})\n".format(xmax, r_ext* 0.70711, -z)) #3
    f.write("    ({} {} {})\n".format(r_ext* 0.70711, r_ext* 0.70711, -z)) #4
    f.write("    ({} {} {})\n".format(r_int* 0.70711, r_int* 0.70711, -z)) #5
    f.write("    ({} {} {})\n".format(xmax, ymax, -z)) #6 
    f.write("    ({} {} {})\n".format(r_ext* 0.70711, ymax, -z)) #7
    f.write("    ({} {} {})\n".format(x_orig, ymax, -z)) #8
    f.write("    ({} {} {})\n".format(x_orig, r_ext, -z)) #9
    f.write("    ({} {} {})\n".format(x_orig, r_int, -z)) #10

    f.write("    ({} {} {})\n".format(-r_int, y_orig, -z)) #11
    f.write("    ({} {} {})\n".format(-r_ext, y_orig, -z)) #12
    f.write("    ({} {} {})\n".format(xmin, y_orig, -z)) #13
    f.write("    ({} {} {})\n".format(xmin, r_ext* 0.70711, -z)) #14
    f.write("    ({} {} {})\n".format(-r_ext* 0.70711, r_ext* 0.70711, -z)) #15
    f.write("    ({} {} {})\n".format(-r_int* 0.70711, r_int* 0.70711, -z)) #16
    f.write("    ({} {} {})\n".format(xmin, ymax, -z)) #17
    f.write("    ({} {} {})\n".format(-r_ext* 0.70711, ymax, -z)) #18

    f.write("    ({} {} {})\n".format(r_int, y_orig, z)) #19
    f.write("    ({} {} {})\n".format(r_ext, y_orig, z)) #20
    f.write("    ({} {} {})\n".format(xmax, y_orig, z)) #21
    f.write("    ({} {} {})\n".format(xmax, r_ext* 0.70711, z)) #22
    f.write("    ({} {} {})\n".format(r_ext* 0.70711, r_ext* 0.70711, z)) #23
    f.write("    ({} {} {})\n".format(r_int* 0.70711, r_int* 0.70711, z)) #24
    f.write("    ({} {} {})\n".format(xmax, ymax, z)) #25 
    f.write("    ({} {} {})\n".format(r_ext* 0.70711, ymax, z)) #26
    f.write("    ({} {} {})\n".format(x_orig, ymax, z)) #27
    f.write("    ({} {} {})\n".format(x_orig, r_ext, z)) #28
    f.write("    ({} {} {})\n".format(x_orig, r_int, z)) #29

    f.write("    ({} {} {})\n".format(-r_int, y_orig, z)) #30
    f.write("    ({} {} {})\n".format(-r_ext, y_orig, z)) #31
    f.write("    ({} {} {})\n".format(xmin, y_orig, z)) #32
    f.write("    ({} {} {})\n".format(xmin, r_ext* 0.70711, z)) #33
    f.write("    ({} {} {})\n".format(-r_ext* 0.70711, r_ext* 0.70711, z)) #34
    f.write("    ({} {} {})\n".format(-r_int* 0.70711, r_int* 0.70711, z)) #35
    f.write("    ({} {} {})\n".format(xmin, ymax, z)) #36
    f.write("    ({} {} {})\n".format(-r_ext* 0.70711, ymax, z)) #37

    f.write(");\n"
            "\n"
            "blocks\n"
            "(\n")
    f.write("    hex (5 4 9 10 24 23 28 29) ({} {} {}) simpleGrading (3 1 1)\n".format(x_cell, y_cell, 1))
    f.write("    hex (0 1 4 5 19 20 23 24) ({} {} {}) simpleGrading (3 1 1)\n".format(x_cell, y_cell, 1))
    f.write("    hex (1 2 3 4 20 21 22 23) ({} {} {}) simpleGrading (10 1 1)\n".format(x_cell2 , y_cell, 1))
    f.write("    hex (4 3 6 7 23 22 25 26) ({} {} {}) simpleGrading (10 0.333 1)\n".format(x_cell2, y_cell2, 1))
    f.write("    hex (9 4 7 8 28 23 26 27) ({} {} {}) simpleGrading (1 0.333 1)\n".format(x_cell, y_cell2, 1))
    f.write("    hex (15 16 10 9 34 35 29 28) ({} {} {}) simpleGrading (0.333 1 1)\n".format(x_cell, y_cell, 1))
    f.write("    hex (12 11 16 15 31 30 35 34) ({} {} {}) simpleGrading (0.333 1 1)\n".format(x_cell, y_cell, 1))
    f.write("    hex (13 12 15 14 32 31 34 33) ({} {} {}) simpleGrading (0.1 1 1)\n".format(x_cell3, y_cell, 1))
    f.write("    hex (14 15 18 17 33 34 37 36) ({} {} {}) simpleGrading (0.1 0.333 1)\n".format(x_cell3, y_cell2, 1))
    f.write("    hex  (15 9 8 18 34 28 27 37) ({} {} {}) simpleGrading (1 0.333 1)\n".format(x_cell, y_cell2, 1))


    f.write(");\n")
    f.write("\n"
            "edges\n"
            "(\n")
    f.write("    arc 0 5 ({} {} {})\n".format(0.866 * r_int, 0.5 * r_int, -z)) 
    f.write("    arc 5 10 ({} {} {})\n".format(0.5 * r_int, 0.866 * r_int, -z))
    f.write("    arc 1 4 ({} {} {})\n".format(0.5 * r_ext, 0.866 * r_ext, -z)) 
    f.write("    arc 4 9 ({} {} {})\n".format(0.5 * r_ext, 0.866 * r_ext, -z))
    f.write("    arc 19 24 ({} {} {})\n".format(0.866 * r_int, 0.5 * r_int, z)) 
    f.write("    arc 24 29 ({} {} {})\n".format(0.5 * r_int, 0.866 * r_int, z))
    f.write("    arc 20 23 ({} {} {})\n".format(0.5 * r_ext, 0.866 * r_ext, z)) 
    f.write("    arc 23 28 ({} {} {})\n".format(0.5 * r_ext, 0.866 * r_ext, z))
    f.write("    arc 11 16 ({} {} {})\n".format(-0.866 * r_int, 0.5 * r_int, -z)) 
    f.write("    arc 16 10 ({} {} {})\n".format(-0.5 * r_int, 0.866 * r_int, -z))
    f.write("    arc 12 15 ({} {} {})\n".format(-0.5 * r_ext, 0.866 * r_ext, -z)) 
    f.write("    arc 15 9 ({} {} {})\n".format(-0.5 * r_ext, 0.866 * r_ext, -z))
    f.write("    arc 30 35 ({} {} {})\n".format(-0.866 * r_int, 0.5 * r_int, z)) 
    f.write("    arc 35 29 ({} {} {})\n".format(-0.5 * r_int, 0.866 * r_int, z))
    f.write("    arc 31 34 ({} {} {})\n".format(-0.5 * r_ext, 0.866 * r_ext, z))
    f.write("    arc 34 28 ({} {} {})\n".format(-0.5 * r_ext, 0.866 * r_ext, z))

    f.write(");\n"
            "\n")
    f.write("boundary\n"
            "(\n"
            "    inlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (14 13 32 33)\n"
            "            (17 14 33 36)\n"
            "        );\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (2 3 22 21)\n"
	    "		 (3 6 25 22)\n"
            "        );\n"
            "    }\n"
            "    top\n"
            "    {\n"
            "        type wall;\n"
            "        faces\n"
            "        (\n"
            "            (7 8 27 26)\n"
            "            (6 7 26 25)\n"
            "            (8 18 37 27)\n"
            "            (18 17 36 37)\n"
            "        );\n"
            "    }\n"
            "    obstacle\n"
            "    {\n"
            "        type wall;\n"
            "        faces\n"
            "        (\n"
            "            (10 5 24 29)\n"
            "            (5 0 19 24)\n"
            "            (16 10 29 35)\n"
            "            (11 16 35 30)\n"
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
parser.add_argument("r_int", help="interior radius")
parser.add_argument("y_max", help="position of the top wall")

args = parser.parse_args()
gen_blockMeshDict( float(args.r_int), float(args.y_max)  )
