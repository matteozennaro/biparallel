#ifndef Py_COREFUNCS_H
#define Py_COREFUNCS_H
#ifdef __cplusplus
extern "C" {
#endif

/* Header file for spammodule */

/* C API functions */
#define PyCore_System_NUM 0
#define PyCore_System_RETURN int
#define PyCore_System_PROTO (const char *command)

/* Total number of C API pointers */
#define PyCore_API_pointers 1


#ifdef CORE_FUNCTIONS_MODULE
/* This section is used when compiling powermodule.c */

static PyCore_System_RETURN PyCore_System PyCore_System_PROTO;

#else
/* This section is used in modules that use spammodule's API */

static void **PyCore_API;

#define PyCore_System \
 (*(PyCore_System_RETURN (*)PyCore_System_PROTO) PyCore_API[PyCore_System_NUM])

/* Return -1 on error, 0 on success.
 * PyCapsule_Import will set an exception if there's an error.
 */
static int
import_core(void)
{
    PyCore_API = (void **)PyCapsule_Import("core._C_API", 0);
    return (PyCore_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* !defined(Py_COREFUNCS_H) */