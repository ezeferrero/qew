#ifndef _COMMON_H_
#define _COMMON_H_

#include <math.h>
#include <sys/time.h> /* struct timeval */

/* DEVICE_FUNCTION declares a function that
   is available in both host and device code */
#ifdef __CUDACC__
#define DEVICE_FUNCTION __host__ __device__
#else
#define DEVICE_FUNCTION
#endif

/* Use C linkage (nvcc is a C++ compiler) */
#ifdef __cplusplus
extern "C" {
#endif


/* A time constant */
#define MICROSEC (1E-6)

/* Ceiling of the division */
static DEVICE_FUNCTION int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

/* Convert (x,y) to a row major representation index */
static DEVICE_FUNCTION int get_index(int x, int y, int stride) {
    return x + y * stride;
}

/* Compare two floating point values */
static DEVICE_FUNCTION int is_different(float a, float b, float epsilon) {
    return fabsf(a - b) >= epsilon;
}

/* Write to a PPM image */
void writePPMbinaryImage(const char * filename, const float * vector);

/*
 * http://www.gnu.org/software/libtool/manual/libc/Elapsed-Time.html
 * Subtract the `struct timeval' values X and Y,
 * storing the result in RESULT.
 * Return 1 if the difference is negative, otherwise 0.
 */
int timeval_subtract (struct timeval * result, struct timeval * x, struct timeval * y);

/* Swap two pointers */
void swap_pointers (void **ptr1, void **ptr2);

#ifdef __cplusplus
}
#endif

#endif /* _COMMON_H_ */
