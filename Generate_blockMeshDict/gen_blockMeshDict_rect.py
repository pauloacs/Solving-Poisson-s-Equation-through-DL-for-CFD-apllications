import argparse


def gen_blockMeshDict(x_cord, x_cord2, y_cord, cell_scale, grading):
    """
    Create a `blockMeshDict` file for the geometry
    """
    grading = grading
    scale_for_cells = cell_scale *10

    scale = 1
    z = 0.05
    x_orig = 0
    y_orig = 0
    x_max = 15
    y_max = 1

    x_cord = x_cord
    x_cord2 = x_cord2
    y_cord = y_cord

    x_cell = int(x_cord2 * scale_for_cells*2)
    x_cell2 = int((x_cord2 - x_cord) * scale_for_cells * 2)
    y_cell = int(y_cord * scale_for_cells*2)

    total_cells_x = x_max * scale_for_cells*2
    total_cells_y = y_max * scale_for_cells *2 + y_cell*0.2

    y_cell2 = int((total_cells_y - y_cell + y_cell) ) #as y_cord gets bigger, the finner the mesh must be in this region, therefore I apply a factor varying with y_cord
    x_cell3 = int(total_cells_x - x_cell - x_cell2)

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

    f.write("    ({} {} {})\n".format(x_orig, y_max/2 + y_cord/2, z)) #24
    f.write("    ({} {} {})\n".format(x_orig, y_max/2 + y_cord/2, -z)) #25
    f.write("    ({} {} {})\n".format(x_cord, y_max/2 + y_cord/2, z)) #26
    f.write("    ({} {} {})\n".format(x_cord, y_max/2 + y_cord/2, -z)) #27
    f.write("    ({} {} {})\n".format(x_cord2, y_max/2 + y_cord/2, z)) #28
    f.write("    ({} {} {})\n".format(x_cord2, y_max/2 + y_cord/2, -z)) #29
    f.write("    ({} {} {})\n".format(x_max, y_max/2 + y_cord/2, z)) #30
    f.write("    ({} {} {})\n".format(x_max, y_max/2 + y_cord/2, -z)) #31



    f.write(");\n"
            "\n"
            "blocks\n"
            "(\n")
    f.write("    hex (0 1 3 2 8 9 11 10) ({} {} {}) simpleGrading (0.2 1 1)\n".format(x_cell, y_cell, 1))

    f.write("    hex (2 3 27 25 10 11 26 24) ({} {} {}) simpleGrading (0.2 {} 1)\n".format(x_cell, y_cell2 , 1, grading))
    f.write("    hex (25 27 6 5 24 26 14 13) ({} {} {}) simpleGrading (0.2 {} 1)\n".format(x_cell, y_cell2, 1, 1/grading))

    f.write("    hex (3 4 29 27 11 12 28 26) ({} {} {}) simpleGrading (1 {} 1)\n".format( x_cell2 , y_cell2, 1, grading))
    f.write("    hex (27 29 7 6 26 28 15 14) ({} {} {}) simpleGrading (1 {} 1)\n".format(x_cell2 , y_cell2, 1, 1/grading))

    f.write("    hex (4 16 31 29 12 20 30 28) ({} {} {}) simpleGrading (5 {} 1)\n".format( x_cell3 , y_cell2, 1, grading))
    f.write("    hex (29 31 17 7 28 30 21 15) ({} {} {}) simpleGrading (5 {} 1)\n".format(x_cell3 , y_cell2, 1, 1/grading))

    f.write("    hex (18 19 16 4 22 23 20 12) ({} {} {}) simpleGrading (5 1 1)\n".format( x_cell3 , y_cell, 1))


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
            "            (2 10 24 25)\n"
            "            (25 24 13 5)\n"
            "        );\n"
            "    }\n"
            "    outlet\n"
            "    {\n"
            "        type patch;\n"
            "        faces\n"
            "        (\n"
            "            (19 16 20 23)\n"
	    "		 (16 31 30 20)\n"
	    "		 (30 31 17 21)\n"
            "        );\n"
            "    }\n"
            "    top\n"
            "    {\n"
            "        type wall;\n"
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


# Total cell 7500

parser = argparse.ArgumentParser(description="Generating blockMeshDict file for the geometry")
parser.add_argument("x_cord", help="X coordinate of forward step")
parser.add_argument("x_cord2", help="X2 coordinate of forward step")
parser.add_argument("y_cord", help="Y coordinate of forward step")
parser.add_argument("cell_scale", help="define the refinement level")
parser.add_argument("grading", help="grading")

args = parser.parse_args()
gen_blockMeshDict(float(args.x_cord), float(args.x_cord2) , float(args.y_cord), float(args.cell_scale), float(args.grading) )
