#ifndef NUMC_H
#define NUMC_H

#include <vector>
#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <map>
#include <tuple>
#include <cstdint>

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numc {

class Array {
private:
    std::vector<double> data;
    std::vector<size_t> shape;
    size_t ndim;
    size_t size;

    size_t get_index(const std::vector<size_t>& indices) const;
    void compute_shape_product();

public:
    Array();
    Array(const std::vector<size_t>& shape);
    Array(const std::vector<double>& data, const std::vector<size_t>& shape);
    Array(std::initializer_list<std::initializer_list<double>> data);
    
    std::vector<size_t> get_shape() const { return shape; }
    size_t get_ndim() const { return ndim; }
    size_t get_size() const { return size; }
    const std::vector<double>& get_data_ref() const { return data; }
    std::vector<double>& get_data_ref_mutable() { return data; }
    friend Array where(const Array& condition, const Array& x, const Array& y);
    friend Array interp(const Array& x_new, const Array& x, const Array& y);
    friend Array cumtrapz(const Array& y, const Array& x);
    
    Array reshape(const std::vector<size_t>& new_shape) const;
    Array transpose() const;
    Array flatten() const;
    
    double& operator()(const std::vector<size_t>& indices);
    const double& operator()(const std::vector<size_t>& indices) const;
    double& operator[](size_t i);
    const double& operator[](size_t i) const;
    
    Array operator+(const Array& other) const;
    Array operator-(const Array& other) const;
    Array operator*(const Array& other) const;
    Array operator/(const Array& other) const;
    Array operator+(double scalar) const;
    Array operator-(double scalar) const;
    Array operator*(double scalar) const;
    Array operator/(double scalar) const;
    
    Array& operator+=(const Array& other);
    Array& operator-=(const Array& other);
    Array& operator*=(const Array& other);
    Array& operator/=(const Array& other);
    
    bool operator==(const Array& other) const;
    bool operator!=(const Array& other) const;
    
    void print() const;
    void print_shape() const;
    
    static Array zeros(const std::vector<size_t>& shape);
    static Array ones(const std::vector<size_t>& shape);
    static Array arange(double start, double end, double step = 1.0);
    static Array linspace(double start, double end, size_t num);
    static Array full(const std::vector<size_t>& shape, double value);
    static Array identity(size_t n);
    static Array eye(size_t n, size_t m);
    
    double sum() const;
    double mean() const;
    double std_dev() const;
    double var() const;
    double min() const;
    double max() const;
    Array min_along_axis(size_t axis) const;
    Array max_along_axis(size_t axis) const;
    Array sum_along_axis(size_t axis) const;
    Array mean_along_axis(size_t axis) const;
    
    Array dot(const Array& other) const;
    Array matmul(const Array& other) const;
    
    Array sqrt() const;
    Array abs() const;
    Array pow(double n) const;
    Array exp() const;
    Array log() const;
    
    Array concatenate(const Array& other, size_t axis = 0) const;
    std::pair<Array, Array> split(size_t axis, size_t index) const;
    
    Array slice(const std::vector<std::pair<size_t, size_t>>& ranges) const;
    
    std::vector<size_t> argmax() const;
    std::vector<size_t> argmin() const;
    
    void fill(double value);
    void random_normal(double mean = 0.0, double std = 1.0);
    void random_uniform(double low = 0.0, double high = 1.0);
    
    Array sin() const;
    Array cos() const;
    Array tan() const;
    Array asin() const;
    Array acos() const;
    Array atan() const;
    Array sinh() const;
    Array cosh() const;
    Array tanh() const;
    Array asinh() const;
    Array acosh() const;
    Array atanh() const;
    
    Array floor() const;
    Array ceil() const;
    Array round() const;
    Array trunc() const;
    Array fabs() const;
    
    Array sign() const;
    Array square() const;
    Array cbrt() const;
    Array reciprocal() const;
    
    Array log2() const;
    Array log10() const;
    Array log1p() const;
    Array expm1() const;
    
    Array clip(double min_val, double max_val) const;
    Array copy() const;
    Array ravel() const;
    Array squeeze() const;
    Array expand_dims(size_t axis) const;
    
    Array repeat(size_t repeats, size_t axis = 0) const;
    Array tile(const std::vector<size_t>& reps) const;
    
    Array sort(size_t axis = -1) const;
    std::vector<size_t> argsort(size_t axis = -1) const;
    
    bool all() const;
    bool any() const;
    
    Array where(const Array& condition, const Array& other) const;
    std::vector<size_t> nonzero() const;
    Array take(const std::vector<size_t>& indices) const;
    void put(const std::vector<size_t>& indices, const Array& values);
    
    double median() const;
    double percentile(double q) const;
    double quantile(double q) const;
    
    Array unique() const;
    std::pair<Array, Array> unique_with_counts() const;
    
    std::vector<bool> isnan() const;
    std::vector<bool> isinf() const;
    std::vector<bool> isfinite() const;
    
    Array nan_to_num(double nan = 0.0, double posinf = 0.0, double neginf = 0.0) const;
    
    double std_along_axis(size_t axis) const;
    double var_along_axis(size_t axis) const;
    Array prod() const;
    Array prod_along_axis(size_t axis) const;
    double cumsum(size_t i) const;
    Array cumprod(size_t i) const;
    
    Array flip(size_t axis = -1) const;
    Array rot90(int k = 1) const;
    
    bool allclose(const Array& other, double rtol = 1e-05, double atol = 1e-08) const;
    
    std::vector<double> diag() const;
    Array trace() const;
    Array triu(int k = 0) const;
    Array tril(int k = 0) const;
    
    Array pad(const std::vector<std::pair<size_t, size_t>>& pad_width, double constant_value = 0) const;
    
    std::vector<Array> vsplit(size_t indices) const;
    std::vector<Array> hsplit(size_t indices) const;
    std::vector<Array> array_split(size_t indices, size_t axis = 0) const;
    
    static Array stack(const std::vector<Array>& arrays, size_t axis = 0);
    static Array vstack(const std::vector<Array>& arrays);
    static Array hstack(const std::vector<Array>& arrays);
    static Array column_stack(const std::vector<Array>& arrays);
    static Array row_stack(const std::vector<Array>& arrays);
    
    static Array frombuffer(const std::vector<uint8_t>& buffer, const std::vector<size_t>& shape);
    std::vector<uint8_t> tobytes() const;
    
    bool all_equal(const Array& other) const;
    Array angle(double deg = false) const;
    Array unwrap() const;
    
    Array diff(size_t n = 1, size_t axis = -1) const;
    Array ediff1d() const;
    
    double cov(const Array& other) const;
    double corrcoef(const Array& other) const;
    
    Array convolve(const Array& other, const std::string& mode = "full") const;
    Array cross(const Array& other) const;
    Array outer(const Array& other) const;
    
    Array bin_count(const Array& weights, size_t minlength = 0) const;
    Array bincount(const Array& weights, size_t minlength = 0) const;
    
    Array searchsorted(const Array& v, const std::string& side = "left") const;
    
    std::pair<Array, Array> meshgrid() const;
    
    Array gcd(const Array& other) const;
    Array lcm(const Array& other) const;
    
    Array bitwise_and(const Array& other) const;
    Array bitwise_or(const Array& other) const;
    Array bitwise_xor(const Array& other) const;
    Array bitwise_not() const;
    
    Array left_shift(const Array& other) const;
    Array right_shift(const Array& other) const;
    
    bool isin(const Array& test_elements) const;
    
    Array deg2rad() const;
    Array rad2deg() const;
    
    Array heaviside(const Array& h1) const;
    Array nextafter(const Array& to) const;
    
    Array sinc() const;
    Array sinc_interp(const Array& x_new) const;
    
    Array diff_2d(size_t n = 1) const;
    
    std::pair<Array, Array> separate_complex() const;
    Array from_complex(const Array& real, const Array& imag) const;
    
    Array convolve_1d(const Array& other, const std::string& mode) const;
    Array auto_convolve(const std::string& mode) const;
    
    Array fft_1d() const;
    Array ifft_1d() const;
    
    Array autocorr() const;
    Array xcorr(const Array& other) const;
    
    Array moving_average(size_t window) const;
    Array moving_std(size_t window) const;
    
    Array normalize(double a = 0, double b = 1) const;
    Array standardize() const;
    
    Array apply_function(double (*func)(double)) const;
    
    std::string to_string() const;
    std::vector<std::string> to_string_vec() const;
    
    bool is_scalar() const;
    double to_scalar() const;
    
    Array broadcast_to(const std::vector<size_t>& shape) const;
    bool can_broadcast_to(const std::vector<size_t>& shape) const;
    
    std::pair<Array, std::vector<size_t>> broadcast_arrays(const Array& other) const;
    
    Array put_mask(const Array& mask, double value) const;
    Array masked_where(const Array& mask) const;
    
    std::tuple<Array, Array, Array> svd() const;
    std::pair<Array, Array> eig() const;
    Array inv() const;
    double det() const;
    size_t rank() const;
    Array pinv() const;
    Array solve(const Array& b) const;
    
    std::vector<size_t> find_indices(double value) const;
    std::vector<size_t> find_indices_where(const std::string& op, double value) const;
};

Array sin(const Array& a);
Array cos(const Array& a);
Array tan(const Array& a);
Array asin(const Array& a);
Array acos(const Array& a);
Array atan(const Array& a);
Array sinh(const Array& a);
Array cosh(const Array& a);
Array tanh(const Array& a);
Array asinh(const Array& a);
Array acosh(const Array& a);
Array atanh(const Array& a);

Array floor(const Array& a);
Array ceil(const Array& a);
Array round(const Array& a);
Array trunc(const Array& a);
Array fabs(const Array& a);

Array dot(const Array& a, const Array& b);
Array matmul(const Array& a, const Array& b);
Array concatenate(const Array& a, const Array& b, size_t axis = 0);
Array sqrt(const Array& a);
Array abs(const Array& a);
Array exp(const Array& a);
Array log(const Array& a);
Array power(const Array& a, double n);

Array where(const Array& condition, const Array& x, const Array& y);
Array clip(const Array& a, double min_val, double max_val);
Array concatenate_multi(const std::vector<Array>& arrays, size_t axis = 0);
Array stack_arrays(const std::vector<Array>& arrays, size_t axis = 0);
Array stack_v(const std::vector<Array>& arrays);
Array stack_h(const std::vector<Array>& arrays);

Array cross_product(const Array& a, const Array& b);
Array outer_product(const Array& a, const Array& b);
Array convolve_arrays(const Array& a, const Array& b, const std::string& mode);

Array linspace_double(double start, double end, size_t num);
Array logspace(double start, double end, size_t num);
Array geomspace(double start, double end, size_t num);

std::pair<Array, Array> meshgrid(const Array& x, const Array& y);

Array apply_unary_func(const Array& a, double (*func)(double));

double trapz(const Array& y, const Array& x);
double trapz_1d(const Array& y, double dx = 1.0);
Array cumtrapz(const Array& y, const Array& x);

Array interp(const Array& x_new, const Array& x, const Array& y);

}

#endif

