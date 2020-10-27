
# C++ coding style 

## General coding guidelines
* use namespaces
* C++ cast only (no C style cast, i.e. `(int)a`)
* all local variables const (if possible)
* avoid raw memory management (new/delete), except in constructor/destructors
* `#pragma once` instead of header guards
* template function/class definitions in the same header file as the declaration
    - except complete specializations, which cna be implemented in `.cpp` files
* use `.hpp` as the header extensions, and `.cpp` for implementation files 
    - prefer separate implementations in `.cpp` files when possible

## Formatting guidelines
* Capitalized camel-case type names (same for file names)
* `lower_case_with_underscore` function names
* `lower_case_with_underscore` variable names
```
// use
ResourceManager resource_manager
GPI::ResourceManger::get_resource().buffer == get_resource1

// avoid
ResourceManager ResourceManager
GPI::ResourceManger::getResource().Buffer == GetResource1
```

* Includes with `<>` for library headers and `""` for project headers
    * include our own header files with `""`
* Sort includes alphanumerically (with categories other/STL)
    * the files' own header
    * other headers from our own code
    * other libraries
    * STD includes

* Braces on new line for blocks of code 
```
// use this
if (i==0)
{
    return 0;
}

// avoid this
if (i==0) {
    return 0;
}
```

* Use either west/east const
```
int const* const x;
int const y;

const int* const x;
const int y;
```

* Column limit: 90 (both C++ and Python)
* IndentWidth/TabWidth: 2
```
namespace myname
{
  class MyClass
  {
    public:
      Myclass();
  }
}
```

* Pointer/Reference Alignment: Left
```
// use this
int* x;

// avoid this
int *x;
```

* Spaces
    * SpaceAfterTemplateKeyword: true
        `template <typename T>`
    * SpaceBeforeAssignmentOperators: true
    * SpaceBeforeParens in function names: Never
         `x = my_func(x, y);`
    * SpaceInEmptyParentheses: false
    * SpacesInAngles:  false
    * SpacesInParentheses: false
    * SpacesInSquareBrackets: false
    * SpacesInBraces: true `{ }`
         * exception: `T x{};`


