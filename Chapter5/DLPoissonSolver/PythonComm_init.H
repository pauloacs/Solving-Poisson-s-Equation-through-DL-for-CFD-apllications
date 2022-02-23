/*double** input_vals;*/
const int row = num_cells;
const int col = 5; 

const int row_top = num_cells_top;
const int col_top = 2; 

const int row_obst = num_cells_obstacle;
const int col_obst = 2; 

int i, j;

double(*input_vals)[col]{ new double[row][col] };
double(*input_vals_top)[col_top]{ new double[row_top][col_top] };
double(*input_vals_obst)[col_obst]{ new double[row_obst][col_obst] };


forAll(f_U.internalField(), id) // for boundary field use u_.boundaryField()
{
        input_vals[id][0] = U[id].x();
        input_vals[id][1] = U[id].y();
        input_vals[id][2] = C[id].x();
        input_vals[id][3] = C[id].y();
        input_vals[id][4] = p[id];
}

forAll(p.boundaryField()[obst_patchID], id)
{
        input_vals_obst[id][0] = C.boundaryField()[obst_patchID][id].x();
        input_vals_obst[id][1] = C.boundaryField()[obst_patchID][id].y();
}

forAll(p.boundaryField()[top_patchID], id) // for boundary field use u_.boundaryField()
{
        input_vals_top[id][0] = C.boundaryField()[top_patchID][id].x();
        input_vals_top[id][1] = C.boundaryField()[top_patchID][id].y();
}

// Numpy array dimensions
npy_intp dim[] = {num_cells, 5};
npy_intp dim_top[] = {num_cells_top, 2};
npy_intp dim_obstacle[] = {num_cells_obstacle, 2};

// create a new array using 'buffer'
array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, reinterpret_cast<void*>(input_vals));
array_2d_top = PyArray_SimpleNewFromData(2, dim_top, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_top));
array_2d_obst = PyArray_SimpleNewFromData(2, dim_obstacle, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_obst));

PyTuple_SetItem(py_args1, 0, array_2d);
PyTuple_SetItem(py_args1, 1, array_2d_top);
PyTuple_SetItem(py_args1, 2, array_2d_obst);

(void) PyObject_CallObject(py_func1, py_args1);


