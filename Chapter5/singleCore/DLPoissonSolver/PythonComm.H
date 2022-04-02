
forAll(p.internalField(), id) // for boundary field use u_.boundaryField()
{
        input_vals[id][0] = U[id].x();
        input_vals[id][1] = U[id].y();
        input_vals[id][2] = C[id].x();
        input_vals[id][3] = C[id].y();
        input_vals[id][4] = p[id]; // passing also the pressure from previous t to correct some regions
}

// Numpy array dimensions
npy_intp dim[] = {num_cells, 5};

// create a new array using 'buffer'
array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, reinterpret_cast<void*>(input_vals));

PyTuple_SetItem(py_args, 0, array_2d);

// Call python function

pValue = (PyArrayObject*)PyObject_CallObject(py_func, py_args); //Casting to PyArrayObject

forAll (p.internalField(),id)
{
         // Here we assume that pValue has a numpy array of dimension 2 with 1 column only
         p[id] = *((double*)PyArray_GETPTR2(pValue , id , 0));
         out[id] = p[id];
}

