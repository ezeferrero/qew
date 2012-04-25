#ifndef _SETTINGS_H_
#define _SETTINGS_H_

/* Problem size */
#ifndef N /* default: 1000^2 cells */
#define N 1000
#endif

/* Hot temperature */
#ifndef HOT /* default: 1000.0f degees */
#define HOT (1.0f)
#endif

/* Maximum difference */
#ifndef EPSILON /* default: 0.01f */
#define EPSILON (0.01f)
#endif

/* or ... Maximum iterations */
#ifndef MAX_ITERATIONS /* default: 300 times */
#define MAX_ITERATIONS 500
#endif

#endif /* _SETTINGS_H_ */
