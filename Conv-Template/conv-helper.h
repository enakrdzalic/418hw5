/* Header file for a bunch of useful routines to create, read and
   write arrays on which you might want to test your convolution
   routines.
*/

#ifndef CONV_HELPER
#define CONV_HELPER

//--------------------------------------------------------------------------
/* Set a matrix to random values.  To fill an entire matrix, set ldx =
   m.  If ldx > m, you can set a submatrix.

   Note: Assumes column-major addressing, so ldx is added after each
   column.
*/
void rand_float_array(float *x, const uint m, const uint n, const uint ldx,
		      const float lower, const float upper);

//--------------------------------------------------------------------------
/* Set a matrix to a constant value.  To fill an entire matrix, set
   ldx = m.  If ldx > m, you can set a submatrix.

   Note: Assumes column-major addressing, so ldx is added after each
   column.
*/
void const_float_array(float *x, const uint m, const uint n, const uint ldx,
		       const float value);

//--------------------------------------------------------------------------
/* An array which is all value1 except one submatrix which is value2.
*/
void one_spot_array(float *x, const uint m, const uint n, const uint ldx,
		    const uint first_row, const uint first_col,
		    const uint last_row, const uint last_col,
		    const float value1, const float value2);

//--------------------------------------------------------------------------
/* An array which has stripes of two different values.
*/
void stripe_vert_array(float *x, const uint m, const uint n, const uint ldx,
		       const uint width1, const uint width2,
		       const float value1, const float value2);

void stripe_hori_array(float *x, const uint m, const uint n, const uint ldx,
		       const uint height1, const uint height2,
		       const float value1, const float value2);

//--------------------------------------------------------------------------
/* A checkerboard array.
*/
void checkerboard_array(float *x, const uint m, const uint n, const uint ldx,
			const uint height1, const uint width1,
			const uint height2, const uint width2,
			const float value1, const float value2);

//----------------------------------------------------------------------
/* Write a data array out in PGM format ("Portable Greymap") format.

   For readability filename should include the pgm (or ppm) extension,
   but smart image reading files will use the magic number instead.
 */
void write_pgm(const float *data, const uint m, const uint n, 
	       const char *filename);

//----------------------------------------------------------------------
/* Read a data array from PGM format ("Portable Greymap") format.

   Output array is written with float values between 0.0 and 1.0,
   where the latter corresponds to the maximum greyscale value
   reported by the file.
*/
void read_pgm(float **data, uint *p_m, uint *p_n, const char *filename);

#endif
