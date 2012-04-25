#include <assert.h>
#include <math.h>
#include <stdio.h> /* FILE* ops */
#include <stdint.h> /* uint8_t */

//#include "settings.h"
#include "common.h"

/* Write to a PPM image
 * http://en.wikipedia.org/wiki/Portable_Gray_Map
 * http://netpbm.sourceforge.net/doc/ppm.html
 */

/* RGB formulae.
 * http://gnuplot-tricks.blogspot.com/2009/06/comment-on-phonged-surfaces-in-gnuplot.html
 * TODO: cannot find what to do with over/under flows, currently saturate in [0,255]
 */
#define RGBf05 (x*x*x)
#define RGBf07 (sqrt(x))
#define RGBf15 (sin(2*M_PI*x))
#define RGBf21 (3*x)
#define RGBf22 (3*x-1)
#define RGBf23 (3*x-2)
#define RGBf30 (x/0.32-0.78125)
#define RGBf31 (2*x-0.84)
#define RGBf32 (x/0.08-11.5)

#define MAX_COMPONENT_VALUE 255
void writePPMbinaryImage(const char * filename, const cufftComplex * vector)
{
	FILE *f = NULL;
	unsigned int i = 0, j = 0;
	uint8_t RGB[3] = {0,0,0}; /* use 3 first bytes for BGR */ 
	f = fopen(filename, "w");
	assert(f);
	fprintf(f, "P6\n"); /* Portable colormap, binary */
	fprintf(f, "%d %d\n", Lx, Ly); /* Image size */
	fprintf(f, "%d\n", MAX_COMPONENT_VALUE); /* Max component value */
	for (i=0; i<Lx; i++) {
		for (j=0; j<Ly; j++) {
			/* ColorMaps
			 * http://mainline.brynmawr.edu/Courses/cs120/spring2008/Labs/ColorMaps/colorMap.py
			 */
			float x = vector[get_index(i,j,Lx)].x; //HOT; /* [0,1] */
			RGB[0] = (uint8_t) MIN(255, MAX(0, 256*RGBf07));
			RGB[1] = (uint8_t) MIN(255, MAX(0, 256*RGBf05));
			RGB[2] = (uint8_t) MIN(255, MAX(0, 256*RGBf15));
			assert(0<=RGB[0] && RGB[0]<256);
			assert(0<=RGB[1] && RGB[1]<256);
			assert(0<=RGB[2] && RGB[2]<256);
			fwrite((void *)RGB, 3, 1, f);
		}
	}
	fclose(f); f = NULL;
	return;
}

