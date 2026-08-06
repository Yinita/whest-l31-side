void relu_f32(float *x, long long n) {
    for (long long i = 0; i < n; ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

void row_sum_add_f64(const float *x, int rows, int cols, double *acc) {
    for (int i = 0; i < rows; ++i) {
        const float *row = x + (long long)i * cols;
        for (int j = 0; j < cols; ++j) {
            acc[j] += (double)row[j];
        }
    }
}

/* out[i,j] = H[perm[i], j] * signs[j]  (row-major) */
void gather_signed_rows(
    const float *H, int width,
    const int *perm, const float *signs,
    float *out
) {
    for (int i = 0; i < width; ++i) {
        const float *src = H + (long long)perm[i] * width;
        float *dst = out + (long long)i * width;
        for (int j = 0; j < width; ++j) {
            dst[j] = src[j] * signs[j];
        }
    }
}

void negate_copy(const float *src, float *dst, long long n) {
    for (long long i = 0; i < n; ++i) {
        dst[i] = -src[i];
    }
}
