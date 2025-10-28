#include "Numc++.h"
#include <iostream>

using namespace numc;

int main() {
    std::cout << "=== Numc++ Library Demo ===\n\n";
    
    std::cout << "1. Creating arrays:\n";
    Array arr1 = Array::ones({3, 3});
    Array arr2 = Array::zeros({3, 3});
    std::cout << "Ones array:\n";
    arr1.print();
    
    std::cout << "\n2. Arithmetic operations:\n";
    Array sum = arr1 + 2.0;
    std::cout << "Ones + 2:\n";
    sum.print();
    
    std::cout << "\n3. Create range:\n";
    Array arr3 = Array::arange(0, 5, 1);
    std::cout << "Range [0, 5): ";
    arr3.print();
    
    std::cout << "\n4. Statistics:\n";
    std::cout << "Sum: " << arr3.sum() << std::endl;
    std::cout << "Mean: " << arr3.mean() << std::endl;
    std::cout << "Min: " << arr3.min() << std::endl;
    std::cout << "Max: " << arr3.max() << std::endl;
    
    std::cout << "\n5. Matrix operations:\n";
    Array I = Array::identity(3);
    std::cout << "Identity matrix:\n";
    I.print();
    
    Array mat = {{1, 2}, {3, 4}};
    std::cout << "Matrix:\n";
    mat.print();
    std::cout << "Transpose:\n";
    mat.transpose().print();
    
    std::cout << "\n6. Mathematical functions:\n";
    Array values = Array::arange(1, 5, 1);
    std::cout << "Original: ";
    values.print();
    std::cout << "Squared: ";
    values.pow(2).print();
    std::cout << "Square root: ";
    values.sqrt().print();
    
    std::cout << "\n7. Dot product:\n";
    Array vec1 = Array::arange(1, 4, 1);
    Array vec2 = Array::arange(1, 4, 1);
    std::cout << "Vector 1: ";
    vec1.print();
    std::cout << "Vector 2: ";
    vec2.print();
    Array dot_product = vec1.dot(vec2);
    std::cout << "Dot product: ";
    dot_product.print();
    
    return 0;
}



