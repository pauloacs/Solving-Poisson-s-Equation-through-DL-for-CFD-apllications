
Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volVectorField U_non_cons
(
    IOobject
    (
        "U_non_cons",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("U_non_cons", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volScalarField out   // this field is to visualize the output from the surrogate model
(
    IOobject
    (
        "out",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("out", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

#include "createPhi.H"

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, piso.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::momentumTransportModel> turbulence
(
    incompressible::momentumTransportModel::New(U, phi, laminarTransport)
);



/*The following stuff is for Python interoperability*/
PyObject *array_2d;
PyObject *array_2d_top;
PyObject *array_2d_obst;
PyArrayObject *pValue;

Py_Initialize();
PyRun_SimpleString("import sys");
PyRun_SimpleString("sys.path.append(\".\")");

// initialize numpy array library
import_array1(-1);

PyObject *pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
PyObject *pModule = PyImport_Import(pName);
Py_DECREF(pName);

PyObject *py_func = PyObject_GetAttrString(pModule, "py_func");
PyObject *py_args = PyTuple_New(1);
PyObject *py_args_top_bound = PyTuple_New(1);
PyObject *py_args_obst_bound = PyTuple_New(1);

PyObject *py_func1 = PyObject_GetAttrString(pModule, "py_func1");
PyObject *py_args1 = PyTuple_New(3);


cout << 'got here';
Py_DECREF(pModule);

/*// Placeholder to grab data before sending to Python*/
int num_cells = mesh.cells().size();

label top_patchID = mesh.boundaryMesh().findPatchID("top");
int num_cells_top = p.boundaryField()[top_patchID].size();

label obst_patchID = mesh.boundaryMesh().findPatchID("obstacle");
int num_cells_obstacle = p.boundaryField()[obst_patchID].size();

/*Done importing Python functionality*/


#include "createMRF.H"
#include "createFvOptions.H"


/*  Extracting Cell center coordinates */

volVectorField C
(
   IOobject
   (
       "C",
       runTime.timeName(),
       mesh,
       IOobject::NO_READ,
       IOobject::NO_WRITE,
       false
    ),
    mesh.C(),
    calculatedFvPatchScalarField::typeName
);


