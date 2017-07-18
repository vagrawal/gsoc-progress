#include <Python.h>

// int main(int argc, char *argv){
// 	int16_t scores[138];
// 	int frame[1][225];
// 	int i;
// 	int s;

// 	Py_Initialize();
// 	PyObject* myModuleString = PyString_FromString((char*)"runNN");
// 	PyObject* myModule = PyImport_Import(myModuleString);
	
// 	printf("%s\n", PyModule_GetName(myModule));
// 	// pDict is a borrowed reference 
//     PyObject* pDict = PyModule_GetDict(myModule);

//     printf("%s\n", ".....");
//    // pFunc is also a borrowed reference 
//     PyObject* myFunction = PyDict_GetItemString(pDict, (char*)"predictFrame");
// 	printf("%s\n", ".....");
// 	PyObject* args = PyTuple_Pack(1,"mlp1-3x2048-sig-adagrad-bn.h5",frame);
// 	PyObject* myResult = PyObject_CallObject(myFunction, args);
// 	printf("%s\n", "execution finished...");
// 	for(i = 0; i < 138; i++){
// 		s = PyInt_AsLong(PySequence_GetItem(myResult,i));
// 		printf("%d\n", s);
// 		scores[i] = s;
// 	}
// 	Py_Finalize();
// 	return 0;
// }

int main()
{
   // Set PYTHONPATH TO working directory
   setenv("PYTHONPATH",".",1);

   PyObject *pName, *pModule, *pDict, *pFunc, *pValue, *presult;


   // Initialize the Python Interpreter
   Py_Initialize();


   // Build the name object
   pName = PyString_FromString((char*)"arbName");

   // Load the module object
   pModule = PyImport_Import(pName);

   printf("%s\n", ".....");
   // pDict is a borrowed reference 
   pDict = PyModule_GetDict(pModule);
   printf("%s\n", ".....");

   // pFunc is also a borrowed reference 
   pFunc = PyDict_GetItemString(pDict, (char*)"someFunction");
   printf("%s\n", ".....");
   if (PyCallable_Check(pFunc))
   {
       pValue=Py_BuildValue("(z)",(char*)"something");
       PyErr_Print();
       printf("Let's give this a shot!\n");
       presult=PyObject_CallObject(pFunc,pValue);
       PyErr_Print();
   } else 
   {
       PyErr_Print();
   }
   printf("Result is %d\n",PyInt_AsLong(presult));
   Py_DECREF(pValue);

   // Clean up
   Py_DECREF(pModule);
   Py_DECREF(pName);

   // Finish the Python Interpreter
   Py_Finalize();


    return 0;
}