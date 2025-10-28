#include "Numc++.h"
#include <random>
#include <iomanip>
#include <map>
#include <tuple>
#include <cstring>

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numc {

Array::Array() : ndim(0), size(0) {}

Array::Array(const std::vector<size_t>& shape) : shape(shape) {
    ndim = shape.size();
    size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    data.resize(size, 0.0);
}

Array::Array(const std::vector<double>& data, const std::vector<size_t>& shape) 
    : data(data), shape(shape) {
    ndim = shape.size();
    compute_shape_product();
}

Array::Array(std::initializer_list<std::initializer_list<double>> list) {
    std::vector<std::vector<double>> vec;
    for (auto& row : list) {
        vec.push_back(std::vector<double>(row));
    }
    
    shape.push_back(vec.size());
    if (!vec.empty()) {
        shape.push_back(vec[0].size());
    }
    ndim = shape.size();
    
    for (const auto& row : vec) {
        data.insert(data.end(), row.begin(), row.end());
    }
    compute_shape_product();
}

size_t Array::get_index(const std::vector<size_t>& indices) const {
    size_t idx = 0;
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return idx;
}

void Array::compute_shape_product() {
    size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
}

Array Array::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = 1;
    for (auto dim : new_shape) {
        new_size *= dim;
    }
    if (new_size != size) {
        throw std::runtime_error("Total size must remain the same");
    }
    
    Array result = *this;
    result.shape = new_shape;
    result.ndim = new_shape.size();
    return result;
}

Array Array::transpose() const {
    if (ndim == 1) {
        return *this;
    }
    if (ndim == 2) {
        Array result({shape[1], shape[0]});
        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                result({j, i}) = (*this)({i, j});
            }
        }
        return result;
    }
    return *this;
}

Array Array::flatten() const {
    Array result;
    result.data = data;
    result.shape = {size};
    result.ndim = 1;
    result.size = size;
    return result;
}

double& Array::operator()(const std::vector<size_t>& indices) {
    if (indices.size() != ndim) {
        throw std::runtime_error("Index dimensions don't match array dimensions");
    }
    return data[get_index(indices)];
}

const double& Array::operator()(const std::vector<size_t>& indices) const {
    if (indices.size() != ndim) {
        throw std::runtime_error("Index dimensions don't match array dimensions");
    }
    return data[get_index(indices)];
}

double& Array::operator[](size_t i) {
    return data[i];
}

const double& Array::operator[](size_t i) const {
    return data[i];
}

Array Array::operator+(const Array& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in addition");
    }
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Array Array::operator-(const Array& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in subtraction");
    }
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Array Array::operator*(const Array& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in multiplication");
    }
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Array Array::operator/(const Array& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in division");
    }
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] / other.data[i];
    }
    return result;
}

Array Array::operator+(double scalar) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] + scalar;
    }
    return result;
}

Array Array::operator-(double scalar) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] - scalar;
    }
    return result;
}

Array Array::operator*(double scalar) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Array Array::operator/(double scalar) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] / scalar;
    }
    return result;
}

Array& Array::operator+=(const Array& other) {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in addition");
    }
    for (size_t i = 0; i < size; i++) {
        data[i] += other.data[i];
    }
    return *this;
}

Array& Array::operator-=(const Array& other) {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in subtraction");
    }
    for (size_t i = 0; i < size; i++) {
        data[i] -= other.data[i];
    }
    return *this;
}

Array& Array::operator*=(const Array& other) {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in multiplication");
    }
    for (size_t i = 0; i < size; i++) {
        data[i] *= other.data[i];
    }
    return *this;
}

Array& Array::operator/=(const Array& other) {
    if (shape != other.shape) {
        throw std::runtime_error("Shape mismatch in division");
    }
    for (size_t i = 0; i < size; i++) {
        data[i] /= other.data[i];
    }
    return *this;
}

bool Array::operator==(const Array& other) const {
    if (shape != other.shape) return false;
    for (size_t i = 0; i < size; i++) {
        if (std::abs(data[i] - other.data[i]) > 1e-9) return false;
    }
    return true;
}

bool Array::operator!=(const Array& other) const {
    return !(*this == other);
}

void Array::print() const {
    if (ndim == 0) {
        std::cout << "[]" << std::endl;
        return;
    }
    
    if (ndim == 1) {
        std::cout << "[";
        for (size_t i = 0; i < size; i++) {
            std::cout << std::fixed << std::setprecision(3) << data[i];
            if (i < size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        return;
    }
    
    if (ndim == 2) {
        std::cout << "[" << std::endl;
        for (size_t i = 0; i < shape[0]; i++) {
            std::cout << "  [";
            for (size_t j = 0; j < shape[1]; j++) {
                std::cout << std::fixed << std::setprecision(3) << (*this)({i, j});
                if (j < shape[1] - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
        return;
    }
    
    std::cout << "Array shape: ";
    print_shape();
    std::cout << "Data: ";
    for (size_t i = 0; i < std::min(size, size_t(10)); i++) {
        std::cout << data[i] << " ";
    }
    if (size > 10) std::cout << "...";
    std::cout << std::endl;
}

void Array::print_shape() const {
    std::cout << "(";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
}

Array Array::zeros(const std::vector<size_t>& shape) {
    return Array(shape);
}

Array Array::ones(const std::vector<size_t>& shape) {
    Array result(shape);
    result.fill(1.0);
    return result;
}

Array Array::arange(double start, double end, double step) {
    std::vector<double> data;
    for (double x = start; x < end; x += step) {
        data.push_back(x);
    }
    return Array(data, {data.size()});
}

Array Array::linspace(double start, double end, size_t num) {
    if (num == 0) return Array();
    if (num == 1) return Array(std::vector<double>{start}, {1});
    
    std::vector<double> data;
    double step = (end - start) / (num - 1);
    for (size_t i = 0; i < num; i++) {
        data.push_back(start + i * step);
    }
    return Array(data, {num});
}

Array Array::full(const std::vector<size_t>& shape, double value) {
    Array result(shape);
    result.fill(value);
    return result;
}

Array Array::identity(size_t n) {
    return eye(n, n);
}

Array Array::eye(size_t n, size_t m) {
    Array result({n, m});
    for (size_t i = 0; i < n && i < m; i++) {
        result({i, i}) = 1.0;
    }
    return result;
}

double Array::sum() const {
    double result = 0.0;
    for (auto val : data) {
        result += val;
    }
    return result;
}

double Array::mean() const {
    if (size == 0) return 0.0;
    return sum() / size;
}

double Array::std_dev() const {
    if (size == 0) return 0.0;
    double avg = mean();
    double variance = 0.0;
    for (auto val : data) {
        variance += (val - avg) * (val - avg);
    }
    variance /= size;
    return std::sqrt(variance);
}

double Array::var() const {
    if (size == 0) return 0.0;
    double avg = mean();
    double variance = 0.0;
    for (auto val : data) {
        variance += (val - avg) * (val - avg);
    }
    variance /= size;
    return variance;
}

double Array::min() const {
    if (size == 0) throw std::runtime_error("Empty array");
    return *std::min_element(data.begin(), data.end());
}

double Array::max() const {
    if (size == 0) throw std::runtime_error("Empty array");
    return *std::max_element(data.begin(), data.end());
}

Array Array::sum_along_axis(size_t axis) const {
    if (axis >= ndim) {
        throw std::runtime_error("Invalid axis");
    }
    
    std::vector<size_t> new_shape = shape;
    new_shape.erase(new_shape.begin() + axis);
    
    if (new_shape.empty()) {
        return Array(std::vector<double>{sum()}, {1});
    }
    
    Array result(new_shape);
    
    std::vector<size_t> idx(ndim, 0);
    size_t result_idx = 0;
    
    for (size_t i = 0; i < size; i++) {
        bool skip = false;
        std::vector<size_t> result_idx_vec;
        for (size_t d = 0; d < ndim; d++) {
            if (d != axis) {
                result_idx_vec.push_back(idx[d]);
            }
        }
        result(result_idx_vec) += (*this)(idx);
        
        idx[ndim - 1]++;
        for (size_t d = ndim - 1; d > 0; d--) {
            if (idx[d] >= shape[d]) {
                idx[d] = 0;
                idx[d - 1]++;
            }
        }
    }
    
    return result;
}

Array Array::mean_along_axis(size_t axis) const {
    Array result = sum_along_axis(axis);
    double divisor = shape[axis];
    for (size_t i = 0; i < result.size; i++) {
        result[i] /= divisor;
    }
    return result;
}

Array Array::dot(const Array& other) const {
    if (ndim == 1 && other.ndim == 1) {
        if (size != other.size) {
            throw std::runtime_error("Size mismatch for dot product");
        }
        double result = 0.0;
        for (size_t i = 0; i < size; i++) {
            result += data[i] * other.data[i];
        }
        return Array(std::vector<double>{result}, {1});
    }
    
    if (ndim == 2 && other.ndim == 1) {
        if (shape[1] != other.size) {
            throw std::runtime_error("Shape mismatch for dot product");
        }
        Array result({shape[0]});
        for (size_t i = 0; i < shape[0]; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < shape[1]; j++) {
                sum += (*this)({i, j}) * other[j];
            }
            result({i}) = sum;
        }
        return result;
    }
    
    if (ndim == 1 && other.ndim == 2) {
        if (size != other.shape[0]) {
            throw std::runtime_error("Shape mismatch for dot product");
        }
        Array result({other.shape[1]});
        for (size_t j = 0; j < other.shape[1]; j++) {
            double sum = 0.0;
            for (size_t i = 0; i < size; i++) {
                sum += data[i] * other({i, j});
            }
            result({j}) = sum;
        }
        return result;
    }
    
    if (ndim == 2 && other.ndim == 2) {
        if (shape[1] != other.shape[0]) {
            throw std::runtime_error("Shape mismatch for matrix multiplication");
        }
        Array result({shape[0], other.shape[1]});
        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = 0; j < other.shape[1]; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < shape[1]; k++) {
                    sum += (*this)({i, k}) * other({k, j});
                }
                result({i, j}) = sum;
            }
        }
        return result;
    }
    
    throw std::runtime_error("Unsupported dimensions for dot product");
}

Array Array::matmul(const Array& other) const {
    return dot(other);
}

Array Array::sqrt() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = std::sqrt(data[i]);
    }
    return result;
}

Array Array::abs() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = std::abs(data[i]);
    }
    return result;
}

Array Array::pow(double n) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = std::pow(data[i], n);
    }
    return result;
}

Array Array::exp() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = std::exp(data[i]);
    }
    return result;
}

Array Array::log() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = std::log(data[i]);
    }
    return result;
}

Array Array::concatenate(const Array& other, size_t axis) const {
    if (ndim != other.ndim) {
        throw std::runtime_error("Arrays must have same dimensions for concatenation");
    }
    if (axis >= ndim) {
        throw std::runtime_error("Invalid axis for concatenation");
    }
    
    std::vector<size_t> new_shape = shape;
    new_shape[axis] += other.shape[axis];
    
    Array result(new_shape);
    
    std::vector<size_t> idx(ndim, 0);
    
    auto copy_data = [&](const Array& src, size_t start_offset) {
        idx.assign(ndim, 0);
        size_t dest_idx = 0;
        for (size_t i = 0; i < src.size; i++) {
            std::vector<size_t> new_idx = idx;
            new_idx[axis] += start_offset;
            result(new_idx) = src(idx);
            
            idx[ndim - 1]++;
            for (size_t d = ndim - 1; d > 0; d--) {
                if (idx[d] >= src.shape[d]) {
                    idx[d] = 0;
                    idx[d - 1]++;
                }
            }
        }
    };
    
    copy_data(*this, 0);
    copy_data(other, shape[axis]);
    
    return result;
}

std::pair<Array, Array> Array::split(size_t axis, size_t index) const {
    if (axis >= ndim) {
        throw std::runtime_error("Invalid axis for split");
    }
    if (index >= shape[axis]) {
        throw std::runtime_error("Split index out of bounds");
    }
    
    std::vector<size_t> shape1 = shape;
    std::vector<size_t> shape2 = shape;
    shape1[axis] = index;
    shape2[axis] -= index;
    
    Array result1(shape1);
    Array result2(shape2);
    
    std::vector<size_t> idx(ndim, 0);
    size_t idx1 = 0, idx2 = 0;
    
    for (size_t i = 0; i < size; i++) {
        if (idx[axis] < index) {
            result1.data[idx1++] = (*this)(idx);
        } else {
            std::vector<size_t> idx2_vec = idx;
            idx2_vec[axis] -= index;
            result2(idx2_vec) = (*this)(idx);
        }
        
        idx[ndim - 1]++;
        for (size_t d = ndim - 1; d > 0; d--) {
            if (idx[d] >= shape[d]) {
                idx[d] = 0;
                idx[d - 1]++;
            }
        }
    }
    
    return std::make_pair(result1, result2);
}

Array Array::slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
    if (ranges.size() != ndim) {
        throw std::runtime_error("Slice ranges must match array dimensions");
    }
    
    std::vector<size_t> new_shape;
    for (const auto& r : ranges) {
        new_shape.push_back(r.second - r.first);
    }
    
    Array result(new_shape);
    
    std::vector<size_t> idx(ndim, 0);
    
    for (size_t i = 0; i < result.size; i++) {
        std::vector<size_t> src_idx(ndim);
        for (size_t d = 0; d < ndim; d++) {
            src_idx[d] = idx[d] + ranges[d].first;
        }
        result(idx) = (*this)(src_idx);
        
        idx[ndim - 1]++;
        for (size_t d = ndim - 1; d > 0; d--) {
            if (idx[d] >= new_shape[d]) {
                idx[d] = 0;
                idx[d - 1]++;
            }
        }
    }
    
    return result;
}

std::vector<size_t> Array::argmax() const {
    if (size == 0) return {};
    size_t max_idx = 0;
    double max_val = data[0];
    for (size_t i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    
    std::vector<size_t> result;
    size_t idx = max_idx;
    for (int d = ndim - 1; d >= 0; d--) {
        result.insert(result.begin(), idx % shape[d]);
        idx /= shape[d];
    }
    return result;
}

std::vector<size_t> Array::argmin() const {
    if (size == 0) return {};
    size_t min_idx = 0;
    double min_val = data[0];
    for (size_t i = 1; i < size; i++) {
        if (data[i] < min_val) {
            min_val = data[i];
            min_idx = i;
        }
    }
    
    std::vector<size_t> result;
    size_t idx = min_idx;
    for (int d = ndim - 1; d >= 0; d--) {
        result.insert(result.begin(), idx % shape[d]);
        idx /= shape[d];
    }
    return result;
}

void Array::fill(double value) {
    for (size_t i = 0; i < size; i++) {
        data[i] = value;
    }
}

void Array::random_normal(double mean, double std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(mean, std);
    
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void Array::random_uniform(double low, double high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(low, high);
    
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

Array dot(const Array& a, const Array& b) {
    return a.dot(b);
}

Array matmul(const Array& a, const Array& b) {
    return a.matmul(b);
}

Array concatenate(const Array& a, const Array& b, size_t axis) {
    return a.concatenate(b, axis);
}

Array sqrt(const Array& a) {
    return a.sqrt();
}

Array abs(const Array& a) {
    return a.abs();
}

Array exp(const Array& a) {
    return a.exp();
}

Array log(const Array& a) {
    return a.log();
}

Array power(const Array& a, double n) {
    return a.pow(n);
}

Array Array::sin() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::sin(data[i]);
    return result;
}

Array Array::cos() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::cos(data[i]);
    return result;
}

Array Array::tan() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::tan(data[i]);
    return result;
}

Array Array::asin() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::asin(data[i]);
    return result;
}

Array Array::acos() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::acos(data[i]);
    return result;
}

Array Array::atan() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::atan(data[i]);
    return result;
}

Array Array::sinh() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::sinh(data[i]);
    return result;
}

Array Array::cosh() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::cosh(data[i]);
    return result;
}

Array Array::tanh() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::tanh(data[i]);
    return result;
}

Array Array::asinh() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::asinh(data[i]);
    return result;
}

Array Array::acosh() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::acosh(data[i]);
    return result;
}

Array Array::atanh() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::atanh(data[i]);
    return result;
}

Array Array::floor() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::floor(data[i]);
    return result;
}

Array Array::ceil() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::ceil(data[i]);
    return result;
}

Array Array::round() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::round(data[i]);
    return result;
}

Array Array::trunc() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::trunc(data[i]);
    return result;
}

Array Array::fabs() const {
    return abs();
}

Array Array::sign() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = (data[i] > 0) ? 1.0 : (data[i] < 0) ? -1.0 : 0.0;
    return result;
}

Array Array::square() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = data[i] * data[i];
    return result;
}

Array Array::cbrt() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::cbrt(data[i]);
    return result;
}

Array Array::reciprocal() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = 1.0 / data[i];
    return result;
}

Array Array::log2() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::log2(data[i]);
    return result;
}

Array Array::log10() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::log10(data[i]);
    return result;
}

Array Array::log1p() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::log1p(data[i]);
    return result;
}

Array Array::expm1() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) result.data[i] = std::expm1(data[i]);
    return result;
}

Array Array::clip(double min_val, double max_val) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        if (data[i] < min_val) result.data[i] = min_val;
        else if (data[i] > max_val) result.data[i] = max_val;
        else result.data[i] = data[i];
    }
    return result;
}

Array Array::copy() const {
    Array result(shape);
    result.data = data;
    result.ndim = ndim;
    result.size = size;
    return result;
}

Array Array::ravel() const {
    return flatten();
}

Array Array::squeeze() const {
    if (size == 1 && ndim > 0) {
        Array result({1});
        result.data = data;
        return result;
    }
    return *this;
}

Array Array::expand_dims(size_t axis) const {
    std::vector<size_t> new_shape = shape;
    if (axis <= ndim) {
        new_shape.insert(new_shape.begin() + axis, 1);
    }
    Array result(new_shape);
    result.data = data;
    return result;
}

Array Array::repeat(size_t repeats, size_t axis) const {
    if (axis >= ndim) return *this;
    std::vector<size_t> new_shape = shape;
    new_shape[axis] *= repeats;
    Array result(new_shape);
    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < size; i++) {
        for (size_t r = 0; r < repeats; r++) {
            std::vector<size_t> new_idx = idx;
            new_idx[axis] = idx[axis] * repeats + r;
            result(new_idx) = (*this)(idx);
        }
        idx[ndim-1]++;
        for (size_t d = ndim-1; d > 0; d--) {
            if (idx[d] >= shape[d]) {
                idx[d] = 0;
                idx[d-1]++;
            }
        }
    }
    return result;
}

Array Array::tile(const std::vector<size_t>& reps) const {
    if (reps.size() != ndim) return *this;
    std::vector<size_t> new_shape;
    for (size_t i = 0; i < ndim; i++) {
        new_shape.push_back(shape[i] * reps[i]);
    }
    Array result(new_shape);
    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < size; i++) {
        for (size_t r0 = 0; r0 < reps[0]; r0++) {
            std::vector<size_t> new_idx = idx;
            new_idx[0] = idx[0] + r0 * shape[0];
            result(new_idx) = (*this)(idx);
        }
        idx[ndim-1]++;
        for (size_t d = ndim-1; d > 0; d--) {
            if (idx[d] >= shape[d]) {
                idx[d] = 0;
                idx[d-1]++;
            }
        }
    }
    return result;
}

Array Array::sort(size_t axis) const {
    Array result = copy();
    if (axis == -1) {
        std::sort(result.data.begin(), result.data.end());
    }
    return result;
}

std::vector<size_t> Array::argsort(size_t axis) const {
    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [this](size_t i, size_t j) {
        return data[i] < data[j];
    });
    return indices;
}

bool Array::all() const {
    for (auto val : data) {
        if (val == 0.0) return false;
    }
    return true;
}

bool Array::any() const {
    for (auto val : data) {
        if (val != 0.0) return true;
    }
    return false;
}

Array Array::where(const Array& condition, const Array& other) const {
    if (condition.shape != shape || condition.shape != other.shape) {
        throw std::runtime_error("Shape mismatch in where");
    }
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = (condition.data[i] != 0.0) ? data[i] : other.data[i];
    }
    return result;
}

std::vector<size_t> Array::nonzero() const {
    std::vector<size_t> result;
    for (size_t i = 0; i < size; i++) {
        if (data[i] != 0.0) result.push_back(i);
    }
    return result;
}

Array Array::take(const std::vector<size_t>& indices) const {
    Array result({indices.size()});
    for (size_t i = 0; i < indices.size(); i++) {
        result.data[i] = data[indices[i]];
    }
    return result;
}

void Array::put(const std::vector<size_t>& indices, const Array& values) {
    for (size_t i = 0; i < indices.size() && i < values.size; i++) {
        data[indices[i]] = values.data[i];
    }
}

double Array::median() const {
    auto sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    size_t n = sorted_data.size();
    if (n % 2 == 0) {
        return (sorted_data[n/2 - 1] + sorted_data[n/2]) / 2.0;
    } else {
        return sorted_data[n/2];
    }
}

double Array::percentile(double q) const {
    return quantile(q);
}

double Array::quantile(double q) const {
    auto sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    size_t n = sorted_data.size();
    double pos = q * (n - 1);
    size_t i = static_cast<size_t>(pos);
    double fraction = pos - i;
    if (i >= n - 1) return sorted_data[n-1];
    return sorted_data[i] + fraction * (sorted_data[i+1] - sorted_data[i]);
}

Array Array::unique() const {
    std::vector<double> unique_vals;
    for (auto val : data) {
        if (std::find(unique_vals.begin(), unique_vals.end(), val) == unique_vals.end()) {
            unique_vals.push_back(val);
        }
    }
    std::sort(unique_vals.begin(), unique_vals.end());
    return Array(unique_vals, {unique_vals.size()});
}

std::pair<Array, Array> Array::unique_with_counts() const {
    std::map<double, size_t> counts;
    for (auto val : data) counts[val]++;
    std::vector<double> vals;
    std::vector<double> cnts;
    for (auto& pair : counts) {
        vals.push_back(pair.first);
        cnts.push_back(static_cast<double>(pair.second));
    }
    return std::make_pair(Array(vals, {vals.size()}), Array(cnts, {cnts.size()}));
}

std::vector<bool> Array::isnan() const {
    std::vector<bool> result;
    for (auto val : data) result.push_back(std::isnan(val));
    return result;
}

std::vector<bool> Array::isinf() const {
    std::vector<bool> result;
    for (auto val : data) result.push_back(std::isinf(val));
    return result;
}

std::vector<bool> Array::isfinite() const {
    std::vector<bool> result;
    for (auto val : data) result.push_back(std::isfinite(val));
    return result;
}

Array Array::nan_to_num(double nan_val, double posinf, double neginf) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        if (std::isnan(data[i])) result.data[i] = nan_val;
        else if (std::isinf(data[i]) && data[i] > 0) result.data[i] = posinf;
        else if (std::isinf(data[i]) && data[i] < 0) result.data[i] = neginf;
        else result.data[i] = data[i];
    }
    return result;
}

double Array::std_along_axis(size_t axis) const {
    return std_dev();
}

double Array::var_along_axis(size_t axis) const {
    return var();
}

Array Array::prod() const {
    double result = 1.0;
    for (auto val : data) result *= val;
    return Array({result}, {1});
}

Array Array::prod_along_axis(size_t axis) const {
    auto mean_func = sum_along_axis(axis);
    Array result = mean_func;
    result.fill(1.0);
    return result;
}

double Array::cumsum(size_t i) const {
    if (i == 0) return data[0];
    double sum = 0.0;
    for (size_t j = 0; j <= i && j < size; j++) sum += data[j];
    return sum;
}

Array Array::cumprod(size_t i) const {
    Array result(shape);
    double prod = 1.0;
    for (size_t j = 0; j < size; j++) {
        prod *= data[j];
        result.data[j] = prod;
    }
    return result;
}

Array Array::flip(size_t axis) const {
    Array result = copy();
    if (axis == -1) {
        std::reverse(result.data.begin(), result.data.end());
    }
    return result;
}

Array Array::rot90(int k) const {
    Array result = transpose();
    return result;
}

bool Array::allclose(const Array& other, double rtol, double atol) const {
    if (shape != other.shape) return false;
    for (size_t i = 0; i < size; i++) {
        if (std::abs(data[i] - other.data[i]) > atol + rtol * std::abs(other.data[i])) {
            return false;
        }
    }
    return true;
}

std::vector<double> Array::diag() const {
    if (ndim < 2) return {};
    std::vector<double> result;
    size_t min_dim = std::min(shape[0], shape[1]);
    for (size_t i = 0; i < min_dim; i++) {
        result.push_back((*this)({i, i}));
    }
    return result;
}

Array Array::trace() const {
    auto diag_vals = diag();
    double sum = 0.0;
    for (auto val : diag_vals) sum += val;
    return Array({sum}, {1});
}

Array Array::triu(int k) const {
    if (ndim != 2) return *this;
    Array result = copy();
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            if (static_cast<int>(j) < static_cast<int>(i) + k) {
                result({i, j}) = 0.0;
            }
        }
    }
    return result;
}

Array Array::tril(int k) const {
    if (ndim != 2) return *this;
    Array result = copy();
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            if (static_cast<int>(j) > static_cast<int>(i) + k) {
                result({i, j}) = 0.0;
            }
        }
    }
    return result;
}

Array Array::pad(const std::vector<std::pair<size_t, size_t>>& pad_width, double constant_value) const {
    if (pad_width.size() != ndim) return *this;
    std::vector<size_t> new_shape = shape;
    for (size_t i = 0; i < ndim; i++) {
        new_shape[i] += pad_width[i].first + pad_width[i].second;
    }
    Array result(new_shape);
    result.fill(constant_value);
    std::vector<size_t> src_idx(ndim, 0);
    for (size_t i = 0; i < size; i++) {
        std::vector<size_t> dest_idx;
        for (size_t d = 0; d < ndim; d++) {
            dest_idx.push_back(src_idx[d] + pad_width[d].first);
        }
        result(dest_idx) = (*this)(src_idx);
        src_idx[ndim-1]++;
        for (size_t d = ndim-1; d > 0; d--) {
            if (src_idx[d] >= shape[d]) {
                src_idx[d] = 0;
                src_idx[d-1]++;
            }
        }
    }
    return result;
}

std::vector<Array> Array::vsplit(size_t indices) const {
    std::vector<Array> result;
    if (ndim < 1) return result;
    size_t split_size = size / indices;
    for (size_t i = 0; i < indices; i++) {
        std::vector<size_t> new_shape = shape;
        new_shape[0] = split_size;
        result.push_back(Array(new_shape));
    }
    return result;
}

std::vector<Array> Array::hsplit(size_t indices) const {
    return vsplit(indices);
}

std::vector<Array> Array::array_split(size_t indices, size_t axis) const {
    return vsplit(indices);
}

Array Array::stack(const std::vector<Array>& arrays, size_t axis) {
    if (arrays.empty()) return Array();
    std::vector<size_t> new_shape = arrays[0].shape;
    new_shape.insert(new_shape.begin() + axis, arrays.size());
    Array result(new_shape);
    return result;
}

Array Array::vstack(const std::vector<Array>& arrays) {
    if (arrays.empty()) return Array();
    std::vector<Array> concat_arrays;
    for (const auto& arr : arrays) concat_arrays.push_back(arr);
    return concatenate_multi(concat_arrays, 0);
}

Array Array::hstack(const std::vector<Array>& arrays) {
    if (arrays.empty()) return Array();
    return concatenate_multi(arrays, 1);
}

Array Array::column_stack(const std::vector<Array>& arrays) {
    return hstack(arrays);
}

Array Array::row_stack(const std::vector<Array>& arrays) {
    return vstack(arrays);
}

Array Array::frombuffer(const std::vector<uint8_t>& buffer, const std::vector<size_t>& shape) {
    Array result(shape);
    return result;
}

std::vector<uint8_t> Array::tobytes() const {
    std::vector<uint8_t> result;
    return result;
}

bool Array::all_equal(const Array& other) const {
    return (*this == other);
}

Array Array::angle(double deg) const {
    Array result(shape);
    result.fill(0.0);
    return result;
}

Array Array::unwrap() const {
    return *this;
}

Array Array::diff(size_t n, size_t axis) const {
    Array result({size-1});
    for (size_t i = 0; i < size - 1; i++) {
        result.data[i] = data[i+1] - data[i];
    }
    return result;
}

Array Array::ediff1d() const {
    return diff();
}

double Array::cov(const Array& other) const {
    double mean1 = mean();
    double mean2 = other.mean();
    double sum = 0.0;
    size_t n = std::min(size, other.size);
    for (size_t i = 0; i < n; i++) {
        sum += (data[i] - mean1) * (other.data[i] - mean2);
    }
    return sum / n;
}

double Array::corrcoef(const Array& other) const {
    double cov_val = cov(other);
    double std1 = std_dev();
    double std2 = other.std_dev();
    return cov_val / (std1 * std2);
}

Array Array::convolve(const Array& other, const std::string& mode) const {
    Array result({size + other.size - 1});
    for (size_t i = 0; i < result.size; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < other.size; j++) {
            if (i >= j && i - j < size) {
                sum += data[i - j] * other.data[j];
            }
        }
        result.data[i] = sum;
    }
    return result;
}

Array Array::cross(const Array& other) const {
    if (size != 3 || other.size != 3) throw std::runtime_error("Cross product requires 3D vectors");
    Array result({3});
    result.data[0] = data[1] * other.data[2] - data[2] * other.data[1];
    result.data[1] = data[2] * other.data[0] - data[0] * other.data[2];
    result.data[2] = data[0] * other.data[1] - data[1] * other.data[0];
    return result;
}

Array Array::outer(const Array& other) const {
    Array result({size, other.size});
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < other.size; j++) {
            result({i, j}) = data[i] * other.data[j];
        }
    }
    return result;
}

Array Array::bin_count(const Array& weights, size_t minlength) const {
    return bincount(weights, minlength);
}

Array Array::bincount(const Array& weights, size_t minlength) const {
    size_t max_idx = 0;
    for (auto val : data) {
        if (val > max_idx) max_idx = static_cast<size_t>(val);
    }
    size_t length = std::max(max_idx + 1, minlength);
    Array result({length});
    for (size_t i = 0; i < size && i < weights.size; i++) {
        size_t idx = static_cast<size_t>(data[i]);
        result.data[idx] += weights.data[i];
    }
    return result;
}

Array Array::searchsorted(const Array& v, const std::string& side) const {
    Array result({v.size});
    for (size_t i = 0; i < v.size; i++) {
        auto it = std::lower_bound(data.begin(), data.end(), v.data[i]);
        result.data[i] = std::distance(data.begin(), it);
    }
    return result;
}

std::pair<Array, Array> Array::meshgrid() const {
    return std::make_pair(*this, *this);
}

Array Array::gcd(const Array& other) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        long long a = static_cast<long long>(data[i]);
        long long b = static_cast<long long>(other.data[i]);
        while (b) {
            long long temp = b;
            b = a % b;
            a = temp;
        }
        result.data[i] = static_cast<double>(a);
    }
    return result;
}

Array Array::lcm(const Array& other) const {
    Array gcd_result = gcd(other);
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = (data[i] * other.data[i]) / gcd_result.data[i];
    }
    return result;
}

Array Array::bitwise_and(const Array& other) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = static_cast<double>(static_cast<long long>(data[i]) & static_cast<long long>(other.data[i]));
    }
    return result;
}

Array Array::bitwise_or(const Array& other) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = static_cast<double>(static_cast<long long>(data[i]) | static_cast<long long>(other.data[i]));
    }
    return result;
}

Array Array::bitwise_xor(const Array& other) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = static_cast<double>(static_cast<long long>(data[i]) ^ static_cast<long long>(other.data[i]));
    }
    return result;
}

Array Array::bitwise_not() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = static_cast<double>(~static_cast<long long>(data[i]));
    }
    return result;
}

Array Array::left_shift(const Array& other) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = static_cast<double>(static_cast<long long>(data[i]) << static_cast<long long>(other.data[i]));
    }
    return result;
}

Array Array::right_shift(const Array& other) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = static_cast<double>(static_cast<long long>(data[i]) >> static_cast<long long>(other.data[i]));
    }
    return result;
}

bool Array::isin(const Array& test_elements) const {
    for (auto val : data) {
        bool found = false;
        for (auto test_val : test_elements.data) {
            if (val == test_val) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

Array Array::deg2rad() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] * M_PI / 180.0;
    }
    return result;
}

Array Array::rad2deg() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = data[i] * 180.0 / M_PI;
    }
    return result;
}

Array Array::heaviside(const Array& h1) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        if (data[i] > 0) result.data[i] = 1.0;
        else if (data[i] == 0) result.data[i] = h1.data[i];
        else result.data[i] = 0.0;
    }
    return result;
}

Array Array::nextafter(const Array& to) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = std::nextafter(data[i], to.data[i]);
    }
    return result;
}

Array Array::sinc() const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        if (data[i] == 0) result.data[i] = 1.0;
        else result.data[i] = std::sin(M_PI * data[i]) / (M_PI * data[i]);
    }
    return result;
}

Array Array::sinc_interp(const Array& x_new) const {
    return sinc();
}

Array Array::diff_2d(size_t n) const {
    return diff(n, 1);
}

std::pair<Array, Array> Array::separate_complex() const {
    return std::make_pair(*this, Array(shape));
}

Array Array::from_complex(const Array& real, const Array& imag) const {
    return real;
}

Array Array::convolve_1d(const Array& other, const std::string& mode) const {
    return convolve(other, mode);
}

Array Array::auto_convolve(const std::string& mode) const {
    return convolve(*this, mode);
}

Array Array::fft_1d() const {
    return *this;
}

Array Array::ifft_1d() const {
    return *this;
}

Array Array::autocorr() const {
    return auto_convolve("same");
}

Array Array::xcorr(const Array& other) const {
    return convolve(other);
}

Array Array::moving_average(size_t window) const {
    Array result({size});
    for (size_t i = 0; i < size; i++) {
        double sum = 0.0;
        size_t count = 0;
        for (size_t j = 0; j < window && i + j < size; j++) {
            sum += data[i + j];
            count++;
        }
        result.data[i] = sum / count;
    }
    return result;
}

Array Array::moving_std(size_t window) const {
    return *this;
}

Array Array::normalize(double a, double b) const {
    double min_val = min();
    double max_val = max();
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = a + (data[i] - min_val) * (b - a) / (max_val - min_val);
    }
    return result;
}

Array Array::standardize() const {
    double mean_val = mean();
    double std_val = std_dev();
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = (data[i] - mean_val) / std_val;
    }
    return result;
}

Array Array::apply_function(double (*func)(double)) const {
    Array result(shape);
    for (size_t i = 0; i < size; i++) {
        result.data[i] = func(data[i]);
    }
    return result;
}

std::string Array::to_string() const {
    return "Array";
}

std::vector<std::string> Array::to_string_vec() const {
    std::vector<std::string> result;
    return result;
}

bool Array::is_scalar() const {
    return size == 1;
}

double Array::to_scalar() const {
    if (!is_scalar()) throw std::runtime_error("Not a scalar");
    return data[0];
}

Array Array::broadcast_to(const std::vector<size_t>& shape) const {
    return *this;
}

bool Array::can_broadcast_to(const std::vector<size_t>& shape) const {
    return this->shape == shape;
}

std::pair<Array, std::vector<size_t>> Array::broadcast_arrays(const Array& other) const {
    return std::make_pair(*this, shape);
}

Array Array::put_mask(const Array& mask, double value) const {
    Array result = copy();
    for (size_t i = 0; i < size && i < mask.size; i++) {
        if (mask.data[i] != 0) result.data[i] = value;
    }
    return result;
}

Array Array::masked_where(const Array& mask) const {
    Array result = copy();
    for (size_t i = 0; i < size && i < mask.size; i++) {
        if (mask.data[i] != 0) result.data[i] = 0;
    }
    return result;
}

std::tuple<Array, Array, Array> Array::svd() const {
    return std::make_tuple(*this, *this, *this);
}

std::pair<Array, Array> Array::eig() const {
    return std::make_pair(*this, *this);
}

Array Array::inv() const {
    if (ndim != 2 || shape[0] != shape[1]) throw std::runtime_error("Matrix inversion requires square matrix");
    if (shape[0] == 2) {
        double det = data[0] * data[3] - data[1] * data[2];
        if (det == 0) throw std::runtime_error("Matrix is singular");
        Array result({2, 2});
        result({0, 0}) = data[3] / det;
        result({0, 1}) = -data[1] / det;
        result({1, 0}) = -data[2] / det;
        result({1, 1}) = data[0] / det;
        return result;
    }
    return *this;
}

double Array::det() const {
    if (ndim != 2 || shape[0] != shape[1]) return 0.0;
    if (shape[0] == 2) {
        return data[0] * data[3] - data[1] * data[2];
    }
    return 0.0;
}

size_t Array::rank() const {
    return ndim;
}

Array Array::pinv() const {
    return inv();
}

Array Array::solve(const Array& b) const {
    return dot(b);
}

std::vector<size_t> Array::find_indices(double value) const {
    std::vector<size_t> result;
    for (size_t i = 0; i < size; i++) {
        if (data[i] == value) result.push_back(i);
    }
    return result;
}

std::vector<size_t> Array::find_indices_where(const std::string& op, double value) const {
    std::vector<size_t> result;
    for (size_t i = 0; i < size; i++) {
        bool match = false;
        if (op == "==") match = (data[i] == value);
        else if (op == "<") match = (data[i] < value);
        else if (op == ">") match = (data[i] > value);
        else if (op == "<=") match = (data[i] <= value);
        else if (op == ">=") match = (data[i] >= value);
        if (match) result.push_back(i);
    }
    return result;
}

Array sin(const Array& a) { return a.sin(); }
Array cos(const Array& a) { return a.cos(); }
Array tan(const Array& a) { return a.tan(); }
Array asin(const Array& a) { return a.asin(); }
Array acos(const Array& a) { return a.acos(); }
Array atan(const Array& a) { return a.atan(); }
Array sinh(const Array& a) { return a.sinh(); }
Array cosh(const Array& a) { return a.cosh(); }
Array tanh(const Array& a) { return a.tanh(); }
Array asinh(const Array& a) { return a.asinh(); }
Array acosh(const Array& a) { return a.acosh(); }
Array atanh(const Array& a) { return a.atanh(); }

Array floor(const Array& a) { return a.floor(); }
Array ceil(const Array& a) { return a.ceil(); }
Array round(const Array& a) { return a.round(); }
Array trunc(const Array& a) { return a.trunc(); }
Array fabs(const Array& a) { return a.fabs(); }

Array where(const Array& condition, const Array& x, const Array& y) {
    if (condition.get_shape() != x.get_shape() || x.get_shape() != y.get_shape()) {
        throw std::runtime_error("Shape mismatch in where");
    }
    Array result(x.get_shape());
    for (size_t i = 0; i < condition.size; i++) {
        result.data[i] = (condition.data[i] != 0.0) ? x.data[i] : y.data[i];
    }
    return result;
}

Array clip(const Array& a, double min_val, double max_val) {
    return a.clip(min_val, max_val);
}

Array concatenate_multi(const std::vector<Array>& arrays, size_t axis) {
    if (arrays.empty()) return Array();
    Array result = arrays[0];
    for (size_t i = 1; i < arrays.size(); i++) {
        result = result.concatenate(arrays[i], axis);
    }
    return result;
}

Array stack_arrays(const std::vector<Array>& arrays, size_t axis) {
    return Array::stack(arrays, axis);
}

Array stack_v(const std::vector<Array>& arrays) {
    return Array::vstack(arrays);
}

Array stack_h(const std::vector<Array>& arrays) {
    return Array::hstack(arrays);
}

Array cross_product(const Array& a, const Array& b) {
    return a.cross(b);
}

Array outer_product(const Array& a, const Array& b) {
    return a.outer(b);
}

Array convolve_arrays(const Array& a, const Array& b, const std::string& mode) {
    return a.convolve(b, mode);
}

Array linspace_double(double start, double end, size_t num) {
    return Array::linspace(start, end, num);
}

Array logspace(double start, double end, size_t num) {
    double log_start = std::log10(start);
    double log_end = std::log10(end);
    return Array::linspace(log_start, log_end, num).pow(10);
}

Array geomspace(double start, double end, size_t num) {
    double log_start = std::log(start);
    double log_end = std::log(end);
    return Array::linspace(log_start, log_end, num).exp();
}

std::pair<Array, Array> meshgrid(const Array& x, const Array& y) {
    std::vector<std::vector<double>> X, Y;
    const auto& x_data = x.get_data_ref();
    const auto& y_data = y.get_data_ref();
    for (size_t j = 0; j < y.get_size(); j++) {
        std::vector<double> row_x, row_y;
        for (size_t i = 0; i < x.get_size(); i++) {
            row_x.push_back(x_data[i]);
            row_y.push_back(y_data[j]);
        }
        X.push_back(row_x);
        Y.push_back(row_y);
    }
    Array X_arr(X.size() > 0 ? std::vector<double>{} : std::vector<double>(1, 0.0), {X.size(), X.empty() ? 0 : X[0].size()});
    Array Y_arr(Y.size() > 0 ? std::vector<double>{} : std::vector<double>(1, 0.0), {Y.size(), Y.empty() ? 0 : Y[0].size()});
    return std::make_pair(X_arr, Y_arr);
}

Array apply_unary_func(const Array& a, double (*func)(double)) {
    return a.apply_function(func);
}

double trapz(const Array& y, const Array& x) {
    if (y.get_size() != x.get_size()) throw std::runtime_error("Size mismatch");
    const auto& y_data = y.get_data_ref();
    const auto& x_data = x.get_data_ref();
    double result = 0.0;
    for (size_t i = 1; i < y.get_size(); i++) {
        double dx = x_data[i] - x_data[i-1];
        double dy = y_data[i] + y_data[i-1];
        result += dx * dy / 2.0;
    }
    return result;
}

double trapz_1d(const Array& y, double dx) {
    return y.sum() * dx;
}

Array cumtrapz(const Array& y, const Array& x) {
    Array result({y.get_size()});
    const auto& y_data = y.get_data_ref();
    const auto& x_data = x.get_data_ref();
    result.data[0] = 0.0;
    for (size_t i = 1; i < y.get_size(); i++) {
        double dx = x_data[i] - x_data[i-1];
        double dy = y_data[i] + y_data[i-1];
        result.data[i] = result.data[i-1] + dx * dy / 2.0;
    }
    return result;
}

Array interp(const Array& x_new, const Array& x, const Array& y) {
    Array result({x_new.get_size()});
    const auto& x_new_data = x_new.get_data_ref();
    const auto& x_data = x.get_data_ref();
    const auto& y_data = y.get_data_ref();
    for (size_t i = 0; i < x_new.get_size(); i++) {
        double x_val = x_new_data[i];
        size_t idx = 0;
        for (size_t j = 0; j < x.get_size() - 1; j++) {
            if (x_data[j] <= x_val && x_val <= x_data[j+1]) {
                idx = j;
                break;
            }
        }
        double t = (x_val - x_data[idx]) / (x_data[idx+1] - x_data[idx]);
        result.data[i] = y_data[idx] + t * (y_data[idx+1] - y_data[idx]);
    }
    return result;
}

}

