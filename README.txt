================================================================================
                    NUMC++ - NUMERICAL COMPUTING LIBRARY FOR C++
                         Complete Function Reference
================================================================================

OVERVIEW:
---------
Numc++ is a C++ library that provides NumPy-like functionality for numerical
computing. It includes array operations, mathematical functions, statistics,
linear algebra, and more.

INSTALLATION (from GitHub):
---------------------------
1) Clone the repository
   git clone https://github.com/your-user/numcpp.git
   cd numcpp/Numc++

2) Build example (requires g++)
   make

3) Run
   ./example   (Linux/macOS)
   example.exe (Windows)

4) Use in your project
   - Copy Numc++.h and Numc++.cpp into your project, or
   - Add as a submodule and include the path in your build system.
   - Compile with: g++ -std=c++14 -O3 your.cpp Numc++.cpp -o your.exe

NAMESPACE:
---------
All functions are in the 'numc' namespace.

================================================================================
                            ARRAY CONSTRUCTION
================================================================================

1. Array()
   - Default constructor, creates an empty array

2. Array(const std::vector<size_t>& shape)
   - Creates an array with specified shape, initialized with zeros
   - Example: Array({3, 4}) creates a 3x4 array

3. Array(const std::vector<double>& data, const std::vector<size_t>& shape)
   - Creates an array from existing data and shape
   - Example: Array({1,2,3,4}, {2,2}) creates a 2x2 matrix

4. Array(std::initializer_list<std::initializer_list<double>> data)
   - Creates an array using initializer list syntax
   - Example: Array({{1,2},{3,4}}) creates a 2x2 array

================================================================================
                          STATIC CONSTRUCTOR FUNCTIONS
================================================================================

5. Array::zeros(const std::vector<size_t>& shape)
   - Creates an array filled with zeros
   - Example: Array::zeros({3, 4})

6. Array::ones(const std::vector<size_t>& shape)
   - Creates an array filled with ones
   - Example: Array::ones({3, 4})

7. Array::arange(double start, double end, double step = 1.0)
   - Creates a 1D array with values from start to end with given step
   - Example: Array::arange(0, 5, 0.5) creates [0, 0.5, 1, 1.5, ..., 4.5]

8. Array::linspace(double start, double end, size_t num)
   - Creates a 1D array with 'num' evenly spaced values between start and end
   - Example: Array::linspace(0, 1, 5) creates [0, 0.25, 0.5, 0.75, 1]

9. Array::full(const std::vector<size_t>& shape, double value)
   - Creates an array filled with specified value
   - Example: Array::full({3, 3}, 5.0)

10. Array::identity(size_t n)
    - Creates an n x n identity matrix (diagonal ones)
    - Example: Array::identity(3)

11. Array::eye(size_t n, size_t m)
    - Creates an n x m matrix with ones on diagonal and zeros elsewhere
    - Example: Array::eye(3, 4)

================================================================================
                          ARRAY PROPERTIES
================================================================================

12. get_shape() -> std::vector<size_t>
    - Returns the shape of the array
    - Example: arr.get_shape() returns {3, 4} for a 3x4 array

13. get_ndim() -> size_t
    - Returns the number of dimensions
    - Example: arr.get_ndim() returns 2 for a 2D array

14. get_size() -> size_t
    - Returns the total number of elements
    - Example: arr.get_size() returns 12 for a 3x4 array

15. get_data() -> std::vector<double>
    - Returns the underlying data vector

================================================================================
                         ARRAY MANIPULATION
================================================================================

16. reshape(const std::vector<size_t>& new_shape) -> Array
    - Reshapes array to new shape (total size must remain same)
    - Example: arr.reshape({2, 6})

17. transpose() -> Array
    - Returns transposed array (swaps rows and columns for 2D)
    - Example: arr.transpose()

18. flatten() -> Array
    - Converts multi-dimensional array to 1D
    - Example: arr.flatten()

19. slice(const std::vector<std::pair<size_t, size_t>>& ranges) -> Array
    - Extracts a sub-array using slicing ranges
    - Example: arr.slice({{0,2}, {1,3}}) for 2D array

================================================================================
                       ARRAY ACCESS & INDEXING
================================================================================

20. operator()(const std::vector<size_t>& indices) -> double&
    - Accesses element at given indices
    - Example: arr({i, j}) for 2D array

21. operator[](size_t i) -> double&
    - Accesses element at linear index
    - Example: arr[i]

================================================================================
                     ARITHMETIC OPERATIONS
================================================================================

Element-wise operations between two arrays (must have same shape):

22. operator+(const Array& other) -> Array
    - Element-wise addition
    - Example: arr1 + arr2

23. operator-(const Array& other) -> Array
    - Element-wise subtraction
    - Example: arr1 - arr2

24. operator*(const Array& other) -> Array
    - Element-wise multiplication
    - Example: arr1 * arr2

25. operator/(const Array& other) -> Array
    - Element-wise division
    - Example: arr1 / arr2

Scalar operations (broadcasts to all elements):

26. operator+(double scalar) -> Array
    - Adds scalar to each element
    - Example: arr + 5.0

27. operator-(double scalar) -> Array
    - Subtracts scalar from each element
    - Example: arr - 5.0

28. operator*(double scalar) -> Array
    - Multiplies each element by scalar
    - Example: arr * 2.0

29. operator/(double scalar) -> Array
    - Divides each element by scalar
    - Example: arr / 2.0

In-place operations:

30. operator+=(const Array& other) -> Array&
    - In-place addition
    - Example: arr += other

31. operator-=(const Array& other) -> Array&
    - In-place subtraction
    - Example: arr -= other

32. operator*=(const Array& other) -> Array&
    - In-place multiplication
    - Example: arr *= other

33. operator/=(const Array& other) -> Array&
    - In-place division
    - Example: arr /= other

Comparison operators:

34. operator==(const Array& other) -> bool
    - Checks if arrays are equal
    - Example: arr1 == arr2

35. operator!=(const Array& other) -> bool
    - Checks if arrays are not equal
    - Example: arr1 != arr2

================================================================================
                        STATISTICAL FUNCTIONS
================================================================================

36. sum() -> double
    - Returns sum of all elements
    - Example: arr.sum()

37. mean() -> double
    - Returns mean (average) of all elements
    - Example: arr.mean()

38. std_dev() -> double
    - Returns standard deviation of all elements
    - Example: arr.std_dev()

39. var() -> double
    - Returns variance of all elements
    - Example: arr.var()

40. min() -> double
    - Returns minimum value
    - Example: arr.min()

41. max() -> double
    - Returns maximum value
    - Example: arr.max()

42. sum_along_axis(size_t axis) -> Array
    - Returns sum along specified axis
    - Example: arr.sum_along_axis(0)

43. mean_along_axis(size_t axis) -> Array
    - Returns mean along specified axis
    - Example: arr.mean_along_axis(0)

44. argmax() -> std::vector<size_t>
    - Returns indices of maximum element
    - Example: arr.argmax()

45. argmin() -> std::vector<size_t>
    - Returns indices of minimum element
    - Example: arr.argmin()

================================================================================
                        MATHEMATICAL FUNCTIONS
================================================================================

Element-wise mathematical functions:

46. sqrt() -> Array
    - Square root of each element
    - Example: arr.sqrt()

47. abs() -> Array
    - Absolute value of each element
    - Example: arr.abs()

48. pow(double n) -> Array
    - Each element raised to power n
    - Example: arr.pow(2.0)

49. exp() -> Array
    - Exponential of each element (e^x)
    - Example: arr.exp()

50. log() -> Array
    - Natural logarithm of each element
    - Example: arr.log()

================================================================================
                      LINEAR ALGEBRA OPERATIONS
================================================================================

51. dot(const Array& other) -> Array
    - Dot product or matrix multiplication
    - Supports: vector-vector, matrix-vector, vector-matrix, matrix-matrix
    - Example: arr1.dot(arr2)

52. matmul(const Array& other) -> Array
    - Matrix multiplication (alias for dot)
    - Example: arr1.matmul(arr2)

================================================================================
                    ARRAY COMBINATION OPERATIONS
================================================================================

53. concatenate(const Array& other, size_t axis = 0) -> Array
    - Concatenates two arrays along specified axis
    - Example: arr1.concatenate(arr2, 0)

54. split(size_t axis, size_t index) -> std::pair<Array, Array>
    - Splits array along specified axis at given index
    - Returns pair of resulting arrays
    - Example: auto [arr1, arr2] = arr.split(0, 2)

================================================================================
                          ARRAY FILLING
================================================================================

55. fill(double value)
    - Fills entire array with specified value
    - Example: arr.fill(3.5)

56. random_normal(double mean = 0.0, double std = 1.0)
    - Fills array with random values from normal distribution
    - Example: arr.random_normal(0, 1)

57. random_uniform(double low = 0.0, double high = 1.0)
    - Fills array with random values from uniform distribution
    - Example: arr.random_uniform(0, 1)

================================================================================
                          OUTPUT FUNCTIONS
================================================================================

58. print()
    - Prints array contents in readable format
    - Example: arr.print()

59. print_shape()
    - Prints array shape
    - Example: arr.print_shape()

================================================================================
                        STANDALONE FUNCTIONS
================================================================================

60. numc::dot(const Array& a, const Array& b) -> Array
    - Standalone dot product function
    - Example: numc::dot(arr1, arr2)

61. numc::matmul(const Array& a, const Array& b) -> Array
    - Standalone matrix multiplication
    - Example: numc::matmul(arr1, arr2)

62. numc::concatenate(const Array& a, const Array& b, size_t axis) -> Array
    - Standalone concatenation function
    - Example: numc::concatenate(arr1, arr2, 0)

63. numc::sqrt(const Array& a) -> Array
    - Standalone square root function
    - Example: numc::sqrt(arr)

64. numc::abs(const Array& a) -> Array
    - Standalone absolute value function
    - Example: numc::abs(arr)

65. numc::exp(const Array& a) -> Array
    - Standalone exponential function
    - Example: numc::exp(arr)

66. numc::log(const Array& a) -> Array
    - Standalone logarithm function
    - Example: numc::log(arr)

67. numc::power(const Array& a, double n) -> Array
    - Standalone power function
    - Example: numc::power(arr, 2.0)

================================================================================
                              USAGE EXAMPLES
================================================================================

BASIC OPERATIONS:
-----------------
using namespace numc;

// Create arrays
Array arr1 = Array::ones({3, 3});
Array arr2 = Array::zeros({3, 3});

// Arithmetic operations
Array sum = arr1 + arr2;
Array diff = arr1 - arr2;
Array prod = arr1 * arr2;
Array quot = arr1 / arr2;

// Scalar operations
Array scaled = arr1 * 2.5;
Array offset = arr1 + 10.0;

STATISTICS:
-----------
Array arr = Array::arange(1, 10, 1);

double total = arr.sum();
double avg = arr.mean();
double stdev = arr.std_dev();
double minimum = arr.min();
double maximum = arr.max();

MATRICES:
---------
Array A = Array::identity(3);
Array B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

Array C = A.matmul(B);
Array D = A.transpose();

MATHEMATICAL FUNCTIONS:
----------------------
Array arr = Array::arange(1, 10, 1);

Array squared = arr.pow(2);
Array root = arr.sqrt();
Array exp_values = arr.exp();
Array log_values = arr.log();

ARRAY CREATION:
--------------
Array z = Array::zeros({3, 4});
Array o = Array::ones({3, 4});
Array r = Array::arange(0, 10, 0.5);
Array l = Array::linspace(0, 1, 100);
Array f = Array::full({5, 5}, 3.14);
Array I = Array::identity(4);

================================================================================
                              END OF REFERENCE
================================================================================



