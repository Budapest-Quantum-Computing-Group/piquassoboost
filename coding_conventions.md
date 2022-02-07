# Piquassoboost coding conventions


## Header files

In general every `.cpp` file has to have a `.h` header file.
Non-header files that are meant for include should be `.inl` files.

A header file should have header guards and include all the files it needs.

### Include guard

General convention for include guard is `<PROJECT>_<PATH>_<FILENAME>_INCLUDED`.
To have unique names the path should be also used in the definition macro.

Example:
we have file `piquassoboost/common/source/dot.h`.
The include guard should look like this:

```
#ifndef PIQUASSOBOOST_COMMON_DOT_H_INCLUDED
#define PIQUASSOBOOST_COMMON_DOT_H_INCLUDED

...
...

#endif // PIQUASSOBOOST_COMMON_DOT_H_INCLUDED
```

The last comment is not mandatory in case of small files.

### Header file includes

If a header file refers to a symbol defined elsewhere, the file should include the header file which has the declaration or definition of the symbol.
Unnecessary includes are being avoided.

Avoid transitivity at includes.

Try to use the following header ordering (the specific parts are to be separated with newlines):
* C standard headers
* C++ standard library headers
* other used library header
* project headers


## General notes

Use namespace of your project everywhere in general.

For variable names, use camelCase naming convention.

One line should contain maximal 80 characters.

### Local variables

Place a local variables in the narrowest scope possible.
Initialize all variable at the declaration if possible.

Think about efficiency if needed.
The objects should not be constructed and destructed many times just in case it is really needed.

### Global variables

Don't use global variables just in case of constexpr constants.


## Functions

Use camelCase naming convention for functions.

Use input and output parameters for the input and output values except if there is one result: in this case use the return value for output value.
Wherever possible, use `const` modifier at parameters.

Functions should be shorter than 30 lines of code.


## Classes

Use classes wherever needed.
But just in case it is really needed (e.g. just for one function not!).
Use meaningful names in PascalCase format.
Member variable and member function namings are the same as variable namings (use camelCase).


