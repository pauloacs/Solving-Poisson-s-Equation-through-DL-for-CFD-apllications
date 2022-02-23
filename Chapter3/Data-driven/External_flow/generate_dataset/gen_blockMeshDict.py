import argparse



def gen_blockMeshDict(x_dim, y_dim):
    """
    Create a `blockMeshDict` file for the geometry
    """

    scale = 1
    z = 0.05
    x_orig = 0
    y_orig = 0
    x_max = 35 * y_dim + x_dim #got 5 y_dim up and 10 y_dim after 
    y_max = 15 * y_dim
    x_cord = 10 * y_dim
    x_cord2 = x_cord + x_dim
    y_cord = y_dim
    x_cell = int(10 * x_cord/y_dim /5)
    x_cell2 = int(10 * x_dim/y_dim)
    x_cell3 = int(10 * 25/4 )
    y_cell = int(10) #number of cells is defined as 10 to y_cell and near the object is uniform


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
    f.write("    ({} {} {})\n".format(x_orig, y_orig, -z)) #0
    f.write("    ({} {} {})\n".format(x_cord, y_orig, -z)) #1
    f.write("    ({} {} {})\n".format(x_orig, y_cord, -z)) #2
    f.write("    ({} {} {})\n".format(x_cord, y_cord, -z)) #3
    f.write("    ({} {} {})\n".format(x_cord2, y_cord, -z)) #4
    f.write("    ({} {} {})\n".format(x_orig, y_max, -z)) #5
    f.write("    ({} {} {})\n".format(x_cord, y_max, -z)) #6 
    f.write("    ({} {} {})\n".format(x_cord2, y_max, -z)) #7
    f.write("    ({} {} {})\n".format(x_orig, y_orig, z)) #8
    f.write("    ({} {} {})\n".format(x_cord, y_orig, z)) #9
    f.write("    ({} {} {})\n".format(x_orig, y_cord, z)) #10
    f.write("    ({} {} {})\n".format(x_cord, y_cord, z)) #11
    f.write("    ({} {} {})\n".format(x_cord2, y_cord, z)) #12
    f.write("    ({} {} {})\n".format(x_orig, y_max, z)) #13
    f.write("    ({} {} {})\n".format(x_cord, y_max, z)) #14
    f.write("    ({} {} {})\n".format(x_cord2, y_max, z)) #15

    f.write("    ({} {} {})\n".format(x_max, y_cord, -z)) #16
    f.write("    ({} {} {})\n".format(x_max, y_max, -z)) #17
    f.write("    ({} {} {})\n".format(x_cord2, y_orig, -z)) #18
    f.write("    ({} {} {})\n".format(x_max, y_orig, -z)) #19
    f.write("    ({} {} {})\n".format(x_max, y_cord, z)) #20
    f.write("    ({} {} {})\n".format(x_max, y_max, z)) #21
    f.write("    ({} {} {})\n".format(x_cord2, y_orig, z)) #22
    f.write("    ({} {} {})\n".format(x_max, y_orig, z)) #23

    f.write(");\n"
            "\n"
            "blocks\n"
            "(\n")
    f.write("    hex (0 1 3 2 8 9 11 10) ({} {} {}) simpleGrading (0.1 1 1)\n".format(x_cell, y_cell, 1))
    f.write("    hex (2 3 6 5 10 11 14 13) ({} {} {}) simpleGrading (0.1 10 1)\n".format(x_cell, y_cell * 4 - y_cell, 1))
    f.write("    hex (3 4 7 6 11 12 15 14) ({} {} {}) simpleGrading (1 10 1)\n".format(x_cell2, y_cell * 4 - y_cell, 1))
    f.write("    hex (4 16 17 7 12 20 21 15) ({} {} {}) simpleGrading (10 10 1)\n".format(x_cell3, y_cell * 4 - y_cell, 1))
    f.write("    hex (18 19 16 4 22 23 20 12) ({} {} {}) simpleGrading (10 1 1)\n".format(x_cell3, y_cell, 1))


    f.write(");\n"
            "\n"
            "edges\n"
            "(\n"
            ");\n"
            "\n"
            "boundary\n"
            "(\n"
            "    inlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (0 8 10 2)\n"
            "            (2 10 13 5)\n"
            "        );\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (19 16 20 23)\n"
	    "		 (16 17 21 20)\n"
            "        );\n"
            "    }\n"
            "    bottom\n"
            "    {\n"
            "        type wall;\n"
            "        faces\n"
            "        (\n"
            "            (0 1 9 8)\n"
            "            (18 19 23 22)\n"
            "        );\n"
            "    }\n"
            "    top\n"
            "    {\n"
            "        type symmetry;\n"
            "        faces\n"
            "        (\n"
            "            (5 13 14 6)\n"
            "            (6 14 15 7)\n"
            "            (17 7 15 21)\n"
            "        );\n"
            "    }\n"
            "    obstacle\n"
            "    {\n"
            "        type wall;\n"
            "        faces\n"
            "        (\n"
            "            (1 3 11 9)\n"
            "            (3 4 12 11)\n"
            "            (4 18 22 12)\n"
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


parser = argparse.ArgumentParser(description="Generating blockMeshDict file for the geometry")
parser.add_argument("x_dim", help="X dimension")
parser.add_argument("y_dim", help="Y dimension")
args = parser.parse_args()
gen_blockMeshDict(float(args.x_dim), float(args.y_dim))
