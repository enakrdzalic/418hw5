#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "cpu-helper.h"

/* Use "plain PGM" (ASCII data) or regular PGM (binary data) for
   output?  Plain PGM can be inspected in a standard text editor (all
   pixel values are written as ASCII integers between 0 and 255),
   while regular PGM will be more compact (1 byte vs 2-4 bytes per
   pixel).  Note that both types can typically be dramatically
   compressed (for example, by zip / gzip).

   Note that this flag is used only to determine file type for
   write_pgm().  The input function read_pgm() can handle either type
   of file.
*/
#define PLAIN_PGM 1

//--------------------------------------------------------------------------
/* Set a matrix to random values.  To fill an entire matrix, set ldx =
   m.  If ldx > m, you can set a submatrix.

   Note: Assumes column-major addressing, so ldx is added after each
   column.
*/
void rand_float_array(float *x, const uint m, const uint n, const uint ldx,
		      const float lower, const float upper) {

  const double gap = upper - lower;

  float *column_vector = x;

  for(uint j = 0; j < n; j++) {
    for(uint i = 0; i < m; i++) {
      const double unit_uniform = (double)rand() / (double)RAND_MAX;
      column_vector[i] = (float)(gap * unit_uniform + lower);
    }
    // Get the address of the start of the next column.
    column_vector = &column_vector[ldx];
  }
}

//--------------------------------------------------------------------------
/* Set a matrix to a constant value.  To fill an entire matrix, set
   ldx = m.  If ldx > m, you can set a submatrix.

   Note: Assumes column-major addressing, so ldx is the number of
   elements between columns.
*/
void const_float_array(float *x, const uint m, const uint n, const uint ldx,
		       const float value) {

  float *column_vector = x;
  for(uint j = 0; j < n; j++) {
    for(uint i = 0; i < m; i++)
      column_vector[i] = value;
    // Get the address of the start of the next column.
    column_vector = &column_vector[ldx];
  }
}

//--------------------------------------------------------------------------
/* An array which is all value1 except one submatrix which is value2.
*/
void one_spot_array(float *x, const uint m, const uint n, const uint ldx,
		    const uint first_row, const uint first_col,
		    const uint last_row, const uint last_col,
		    const float value1, const float value2) {

  // Set the whole array to the background value.
  const_float_array(x, m, n, ldx, value1);
  // Set the subarray to the foreground value.
  const_float_array(&x[IDX2F(first_row, first_col, m, n)],
		    last_row - first_row, last_col - first_col, ldx, value2);
}

//--------------------------------------------------------------------------
/* An array which has stripes of two different values.
*/
void stripe_vert_array(float *x, const uint m, const uint n, const uint ldx,
		       const uint width1, const uint width2,
		       const float value1, const float value2) {

  // Set the whole array to the background value.
  const_float_array(x, m, n, ldx, value1);

  float *column_vector = &x[ldx * width1];
  while(column_vector < &x[ldx * n]) {
      
    // Set the stripe to the foreground value.
    const_float_array(column_vector, m, width2, ldx, value2);

    // Advance to the next foreground stripe.
    column_vector = &column_vector[ldx * (width1 + width2)];
  }
}

void stripe_hori_array(float *x, const uint m, const uint n, const uint ldx,
		       const uint height1, const uint height2,
		       const float value1, const float value2) {

  // Set the whole array to the background value.
  const_float_array(x, m, n, ldx, value1);

  float *row_vector = &x[height1];
  while(row_vector < &x[m]) {
      
    // Set the stripe to the foreground value.
    const_float_array(row_vector, height2, n, ldx, value2);

    // Advance to the next foreground stripe.
    row_vector = &row_vector[height1 + height2];
  }
}


//--------------------------------------------------------------------------
/* A checkerboard array.
*/
void checkerboard_array(float *x, const uint m, const uint n, const uint ldx,
			const uint height1, const uint width1,
			const uint height2, const uint width2,
			const float value1, const float value2) {

  // Create the checkerboard by horizontal stripes of alternating
  // vertical stripes.
  float *row_vector = x;
  while(row_vector < &x[m]) {
      
    // Do the first set of rows.
    stripe_vert_array(row_vector, height1, n, ldx,
		      width1, width2, value1, value2);

    // Advance to the next stripe.
    row_vector = &row_vector[height1];

    if(row_vector < &row_vector[m]) {
      // Do the second set of rows with flipped values.
      stripe_vert_array(row_vector, height2, n, ldx,
			width2, width1, value2, value1);

      // Advance to the next stripe.
      row_vector = &row_vector[height2];
    }
  }
}

//----------------------------------------------------------------------
/* Write a data array out in PGM format ("Portable Greymap") format.

   For readability filename should include the pgm (or ppm) extension,
   but smart image reading files will use the magic number instead.
 */
void write_pgm(const float *data, const uint m, const uint n, 
	       const char *filename) {

  // For consistency with the read_pgm() function, we create a local
  // copy of the input data pointer.
  const float *x = data;

  // We will always use 1 byte values (maximum 256 different greyscale levels).
  const int max_out = 255;

  // Find the maximum and minimum values in the data.  We will scale
  // those to be 0 and maxval.
  float min_data = x[IDX2F(0,0,m,n)];
  float max_data = min_data;
  for(uint j = 0; j < n; j++)
    for(uint i = 0; i < m; i++) {
      float val = x[IDX2F(i,j,m,n)];
      min_data = (val < min_data) ? val : min_data;
      max_data = (val > max_data) ? val : max_data;
    }
  
  const float offset = min_data;
  const float scale = max_out / (max_data - min_data);

  // Open the file.
  FILE *fh;
  assert(fh = fopen(filename, "w"));

  // Format details taken from http://netpbm.sourceforge.net/doc/pgm.html

  // PGM magic number.
  if(PLAIN_PGM)
    fprintf(fh, "P2\n");
  else
    fprintf(fh, "P5\n");
  // Width and height.
  fprintf(fh, "%d %d\n", n, m);
  // Maximum grey value.
  fprintf(fh, "%d\n", max_out);

  for(uint i = 0; i < m; i++) {
    for(uint j = 0; j < n; j++) {
      if(PLAIN_PGM) {
	uint val = round((x[IDX2F(i,j,m,n)] - offset) * scale);
	fprintf(fh, "%d ", val);
      }
      else {
	unsigned char val = round((x[IDX2F(i,j,m,n)] + offset) * scale);
	assert(fwrite(&val, 1, 1, fh) == 1);
      }
    }
    // End of row.
    if(PLAIN_PGM)
      fprintf(fh, "\n");
  }

  fclose(fh);
}

//----------------------------------------------------------------------
/* Read a data array from PGM format ("Portable Greymap") format.

   Output array is allocated in this routine.  Don't forget to free() it!

   Output array is written with float values between 0.0 and 1.0,
   where the latter corresponds to the maximum greyscale value
   reported by the file.
 */
void read_pgm(float **data, uint *p_m, uint *p_n, const char *filename) {

  // Open the file.
  FILE *fh;
  assert(fh = fopen(filename, "r"));

  // Get the magic number.  Make sure it is PGM, and then 
  char magic_number[2];
  fscanf(fh, "%2c", magic_number);

  if((magic_number[0] != 'P') 
     || ((magic_number[1] != '2') && (magic_number[1] != '5'))) {
    fprintf(stderr, "\nPGM file must have magic number \"P5\" or \"P2\", found \"%c%c\" instead.\n\n", magic_number[0], magic_number[1]);
    exit(-1);
  }

  // Image width, height, maximum greyscale value.
  uint m, n, max_in_int;
  fscanf(fh, "%u %u %u ", &n, &m, &max_in_int);
  const double  max_in_double = (double)max_in_int;

  // Allocate space for the image data.
  float *x;
  NewArrayCPU(float, m*n, x);

  // Read the image data.
  for(uint i = 0; i < m; i++) {
    for(uint j = 0; j < n; j++) {
      if(magic_number[1] == '2') {
	// Plain PGM: read ASCII text.
	uint val;
	fscanf(fh, "%u ", &val);
	x[IDX2F(i,j,m,n)] = (double)val / max_in_double;
      }
      else {
	// Regular PGM: read binary values.
	if(max_in_int < 256) {
	  // One byte values.
	  unsigned char val;
	  assert(fread(&val, 1, 1, fh) == 1);
	  x[IDX2F(i,j,m,n)] = (double)val / max_in_double;
	}	  
	else {
	  // Two byte values.
	  unsigned short val;
	  assert(fread(&val, 2, 1, fh) == 1);
	  x[IDX2F(i,j,m,n)] = (double)val / max_in_double;
	}	  
      }
    }
  }

  fclose(fh);

  *data = x;
  *p_m = m;
  *p_n = n;
}
