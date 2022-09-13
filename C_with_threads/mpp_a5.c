/*
Course work for 'Multiprocessor Programming' - spring 2021
Task: Stereo disparity thread implementation
By: Tiia Leinonen
*/

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <math.h>


#include <windows.h>	    //pthread does not work with Windows, use Windows' own thread functionality
#include <process.h>	   // _beginthread, _endthread

unsigned part = 0;

#define NUM_THREADS 4    //4 threads in the beginning for testing

struct funcparams { unsigned char* inputimage; unsigned width; unsigned height; unsigned char* outputimage; };
typedef struct funcparams Params;

struct zfuncparams { unsigned char* inputimage;  unsigned char* inputimage2; unsigned width; unsigned height; unsigned char* outputimage; unsigned winsize; unsigned order; };
typedef struct zfuncparams ParamsZ;

/* Helper functions for zncc */

float calc_average(unsigned char* image, unsigned width, unsigned height, unsigned x, unsigned y, unsigned d, int sign, unsigned winsize) 
{
	float mean, sum;
	unsigned fx, fy;

	
	mean = 0;
	sum = 0;
	for (fy = 0; fy < winsize; fy += 1)  
	{
		for (fx = 0; fx < winsize; fx += 1)
		{
			if (sign == 0)
			{
				sum += image[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))];
			}
			else 
			{
				sum += image[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))];
			}
		}
	}
	mean = sum / (winsize*winsize);
	
	return mean;
}

float calc_stdev(unsigned char* image, unsigned width, unsigned height, unsigned x, unsigned y, unsigned d, int sign, unsigned winsize)
{
	float avg, sum, stdev;
	unsigned fx, fy;

	avg = calc_average(image, width, height, x, y, d, sign, winsize);

	sum = 0;
	for (fy = 0; fy < winsize; fy += 1) 
	{
		for (fx = 0; fx < winsize; fx += 1)
		{
			if (sign == 0)
			{
				sum += pow((image[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - avg), 2);
			}
			else 
			{
				sum += pow((image[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - avg), 2);
			}
		}
	}
	stdev = pow((sum/ (winsize * winsize)), 0.5);

	return stdev;
}

unsigned __stdcall mean_filter(ParamsZ* params)   //takes in an image, returns it filtered
{
	unsigned width = params->width;
	unsigned height = params->height;
	unsigned winsize = params->winsize;
	unsigned char* image = params->inputimage;
	unsigned char* result_image = params->outputimage;

	unsigned x, y, fy, fx;
	size_t jump = 1;

	unsigned threadpart = part++;   //image will be processed in NUM_THREADS parts
	if (threadpart == NUM_THREADS - 1)   //set global var back to 0 for next function
	{
		part = 0;
	}

	for (y = 0; y + jump - 1 < height; y += jump)    //jump because one rgba value takes 4 numbers
	{
		for (x = (threadpart * width) / NUM_THREADS; x + jump - 1 < ((threadpart + 1) * (width)) / NUM_THREADS; x += jump)
		{
			unsigned char mean;

			unsigned sum = 0;

			if (x < ((winsize - 1) / 2) || y < ((winsize - 1) / 2) || x > width - ((winsize - 1) / 2) - 1 || y > height - ((winsize - 1) / 2) - 1)
			{
				mean = image[4 * y * width + 4 * x];
			}
			else
			{
				mean = 0;
				sum = 0;
				for (fy = 0; fy < winsize; fy += 1)  
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						sum += image[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - ((winsize - 1) / 2))];
					}
				}
				mean = sum / (winsize*winsize);
			}

			/* Save the mean value as the new gray value */
			result_image[4 * y * width + 4 * x + 0] = 250 / 65 * (mean - 65) + 250;
			result_image[4 * y * width + 4 * x + 1] = 250 / 65 * (mean - 65) + 250;
			result_image[4 * y * width + 4 * x + 2] = 250 / 65 * (mean - 65) + 250;
			result_image[4 * y * width + 4 * x + 3] = 255;

		}
	}

	return 0;
}

/* The wanted functions: */

unsigned char* ReadImage(const char* imgname, unsigned* width, unsigned* height)
{ /* Reads and returns the input image in RGBA format.*/

	clock_t st, et;
	double took = 0;

	unsigned error;
	unsigned char* image = 0;

	/* Load the image file and measure the time it takes.*/
	st = clock();
	error = lodepng_decode32_file(&image, width, height, imgname);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;

	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	else printf("Image file %s loaded successfully in %f ms!\n", imgname, took);

	return image;

}

unsigned __stdcall ResizeImage(Params* params)
{ /* Resizes the image down by scale 4 */

	unsigned x, y;
	unsigned scale = 4;
	size_t jump = 1;

	unsigned width = params->width;
	unsigned height = params->height;
	unsigned char* image = params->inputimage;
	unsigned char* result_image = params->outputimage;

	unsigned newwidth = width / scale;
	unsigned newheight = height / scale;

	unsigned threadpart = part++;   //image will be processed in NUM_THREADS parts
	if (threadpart == NUM_THREADS - 1)   //set global var back to 0 for next function
	{
		part = 0;
	}

	//printf("Part %d, threadid %u\n", threadpart, GetCurrentThreadId());
	//for (unsigned i = (threadpart*(width*height)/NUM_THREADS); i < ((threadpart + 1) * (width * height) / NUM_THREADS); i++)

	//process only few columns per thread
	/* Get RGBA components */
	for (y = 0; y + jump - 1 < newheight; y += jump)
	{
		
		for (x = (threadpart * newwidth) / NUM_THREADS; x + jump - 1 < ((threadpart + 1) * (newwidth)) / NUM_THREADS; x += jump)
		{
			unsigned char r, g, b, a;

			r = image[scale * 4 * y * width + scale * 4 * x + 0];   /*red*/
			g = image[scale * 4 * y * width + scale * 4 * x + 1];   /*green*/
			b = image[scale * 4 * y * width + scale * 4 * x + 2];   /*blue*/
			a = image[scale * 4 * y * width + scale * 4 * x + 3];   /*alpha*/

			/* Set same values but every 4th value*/
			result_image[4 * y * newwidth + 4 * x + 0] = r;
			result_image[4 * y * newwidth + 4 * x + 1] = g;
			result_image[4 * y * newwidth + 4 * x + 2] = b;
			result_image[4 * y * newwidth + 4 * x + 3] = a;

		}
	}
	
	return 0;
}

unsigned __stdcall GrayScaleImage(Params* params)
{
	unsigned x, y;
	size_t jump = 1;

	unsigned width = params->width;
	unsigned height = params->height;
	unsigned char* image = params->inputimage;
	unsigned char* result_image = params->outputimage;

	unsigned threadpart = part++;
	if (threadpart == NUM_THREADS - 1)   //set global var back to 0 for next function
	{
		part = 0;
	}

	/* Get RGBA components */
	for (y = 0; y + jump - 1 < height; y += jump) 
	{
		//for (x = 0; x + jump - 1 < width; x += jump)
		for (x = (threadpart * width) / NUM_THREADS; x + jump - 1 < ((threadpart + 1) * (width)) / NUM_THREADS; x += jump)
		{
			unsigned char r, g, b, a, gray;

			r = image[4 * y * width + 4 * x + 0];   /*red*/
			g = image[4 * y * width + 4 * x + 1];   /*green*/
			b = image[4 * y * width + 4 * x + 2];   /*blue*/
			a = image[4 * y * width + 4 * x + 3];   /*alpha*/

			gray = (r * 0.30 + g * 0.59 + b * 0.11);

			result_image[4 * y * width + 4 * x + 0] = gray;
			result_image[4 * y * width + 4 * x + 1] = gray;
			result_image[4 * y * width + 4 * x + 2] = gray;
			result_image[4 * y * width + 4 * x + 3] = a;    //set same alpha value

		}
	}

	return 0;
}

float CalcZNCC(unsigned char* image, unsigned char* image2, unsigned width, unsigned height, unsigned x, unsigned y, unsigned d, int sign, unsigned winsize)
{ /*Zero Normalised Cross Correlation for 5 x 5 window, or 2-pixel range around processed pixel. Apply to grayscaled images. */

	//needs to calculate average and standard deviation for zncc
	float avg1, avg2, stdev1, stdev2, sum, result;
	unsigned fx, fy;

	//disparity value applied to right image only
	avg1 = calc_average(image, width, height, x, y, 0, sign, winsize);  
	avg2 = calc_average(image2, width, height, x, y, d, sign, winsize);

	stdev1 = calc_stdev(image, width, height, x, y, 0, sign, winsize);
	stdev2 = calc_stdev(image2, width, height, x, y, d, sign, winsize);

	sum = 0;
	for (fy = 0; fy < winsize; fy += 1) 
	{
		for (fx = 0; fx < winsize; fx += 1)
		{
			if (sign == 0) {
				sum += ((image[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - avg1) * (image2[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - avg2));
			}
			else 
			{
				sum += ((image[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - avg1) * (image2[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - avg2));
			}
		}
	}
	result = sum / (stdev1 * stdev2);

	return result;
}

unsigned __stdcall CrossCheck(ParamsZ* params)
{
	unsigned width = params->width;
	unsigned height = params->height;
	unsigned char* image = params->inputimage;
	unsigned char* image2 = params->inputimage2;
	unsigned char* result_image = params->outputimage;

	size_t jump = 1;
	unsigned treshold = 8;  //8 is default one
	unsigned diff;

	unsigned threadpart = part++;   //image will be processed in NUM_THREADS parts
	if (threadpart == NUM_THREADS - 1)   //set global var back to 0 for next function
	{
		part = 0;
	}

	//normalize values newvalue= (max'-min')/(max-min)*(value-max)+max'

	for (unsigned y = 0; y + jump - 1 < height; y += jump)
	{
		for (unsigned x = (threadpart * width) / NUM_THREADS; x + jump - 1 < ((threadpart + 1) * (width)) / NUM_THREADS; x += jump)
		{
			diff = abs(image[4 * y * width + 4 * x + 0] - image2[4 * y * width + 4 * x + 0]);   //grayscale image, all r g b channels have same value
			if (diff > treshold) 
			{
				result_image[4 * y * width + 4 * x + 0] = 0;
				result_image[4 * y * width + 4 * x + 1] = 0;
				result_image[4 * y * width + 4 * x + 2] = 0;
				result_image[4 * y * width + 4 * x + 3] = 255;

			}
			else    //values near same, use values from image here, and normalize them
			{
				result_image[4 * y * width + 4 * x + 0] = image[4 * y * width + 4 * x + 0];
				result_image[4 * y * width + 4 * x + 1] = image[4 * y * width + 4 * x + 1];
				result_image[4 * y * width + 4 * x + 2] = image[4 * y * width + 4 * x + 2];
				result_image[4 * y * width + 4 * x + 3] = image[4 * y * width + 4 * x + 3];
			}

		}
	}

	return 0;
}

unsigned __stdcall OcclusionFill(ParamsZ* params)
{
	unsigned width = params->width;
	unsigned height = params->height;
	unsigned winsize = params->winsize;
	unsigned char* image = params->inputimage;
	unsigned char* result_image = params->outputimage;

	size_t jump = 1;
	unsigned mean, count;

	unsigned threadpart = part++;   //image will be processed in NUM_THREADS parts
	if (threadpart == NUM_THREADS - 1)   //set global var back to 0 for next function
	{
		part = 0;
	}

	for (unsigned y = 0; y + jump - 1 < height; y += jump) 
	{
		for (unsigned x = (threadpart * width) / NUM_THREADS; x + jump - 1 < ((threadpart + 1) * (width)) / NUM_THREADS; x += jump)
		{
			if (x < ((winsize - 1) / 2) || y < ((winsize - 1) / 2) || x > width - ((winsize - 1) / 2) -1 || y > height - ((winsize - 1) / 2) -1)
			{
				mean = image[4 * y * width + 4 * x];
			}
			else {
				if (image[4 * y * width + 4 * x] == 0)  //process the black pixels only
				{
					mean = 0;
					count = 1;

					for (unsigned fy = 0; fy < winsize; fy += 1)  
					{
						for (unsigned fx = 0; fx < winsize; fx += 1)
						{
							//find nearesst non zero value
							if (image[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] > 0)   //exclude the zero value pixels from the equation
							{
								mean = image[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))];
								break;
							}
						}
						if (mean != 0) 
						{
							break;
						}
					}
				}
				else 
				{
					mean = image[4 * y * width + 4 * x];
				}

			}

			/* Save the mean value as the new gray value */
			result_image[4 * y * width + 4 * x + 0] = mean;
			result_image[4 * y * width + 4 * x + 1] = mean;
			result_image[4 * y * width + 4 * x + 2] = mean;
			result_image[4 * y * width + 4 * x + 3] = 255;

		}
	}


	return 0;

}

void WriteImage(const char* output_imgname, unsigned char* image, unsigned width, unsigned height)
{

	clock_t st, et;
	double took = 0;
	unsigned error;

	st = clock();
	error = lodepng_encode32_file(output_imgname, image, width, height);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;

	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	else printf("Image file saved successfully as %s in %f ms!\n", output_imgname, took);

}

unsigned __stdcall DisparityMap(ParamsZ* params) 
{
	unsigned width = params->width;
	unsigned height = params->height;
	unsigned winsize = params->winsize;
	unsigned order = params->order;
	unsigned char* gray0 = params->inputimage;
	unsigned char* gray1 = params->inputimage2;
	unsigned char* disp_image = params->outputimage;

	size_t jump = 1;
	unsigned max_disp = 260 / 4;
	float zncc, cur_max_sum;
	unsigned best_disp;

	unsigned threadpart = part++;   //image will be processed in NUM_THREADS parts
	if (threadpart == NUM_THREADS - 1)   //set global var back to 0 for next function
	{
		part = 0;
	}

	//unsigned char* disp_image0 = malloc(width * height * 4);
	for (unsigned y = 0; y + jump - 1 < height; y += jump)
	{
		for (unsigned x = (threadpart * width) / NUM_THREADS; x + jump - 1 < ((threadpart + 1) * (width)) / NUM_THREADS; x += jump)
		{
			if (x < ((winsize - 1) / 2) || y < ((winsize - 1) / 2) || x > width - ((winsize - 1) / 2) - 1 || y > height - ((winsize - 1) / 2) - 1)
			{
				best_disp = 0;
			}
			else {
				cur_max_sum = 0;
				best_disp = 0;
				for (unsigned d = 0; d < max_disp; d++)
				{
					if (order == 0) //if mapping gray0 to gray1
					{
						if (d > x)   //if we go over border, use x as max disparity value available 
						{
							zncc = CalcZNCC(gray0, gray1, width, height, x, y, x, 0, winsize);
							//if zncc > cur max sum
							if (zncc > cur_max_sum)
							{
								cur_max_sum = zncc;   //update best sum
								best_disp = d;     //update best disparity value
							}
						}
						else
						{
							zncc = CalcZNCC(gray0, gray1, width, height, x, y, d, 0, winsize);
							//if zncc > cur max sum
							if (zncc > cur_max_sum)
							{
								cur_max_sum = zncc;   //update best sum
								best_disp = d;     //update best disparity value
							}
						}
					}
					else //if mapping gray1 to gray0
					{
						if (d > (width - x))   //if we go over border, use width-x as max disparity value available 
						{
							zncc = CalcZNCC(gray1, gray0, width, height, x, y, (width - x), 1, winsize);
							//if zncc > cur max sum
							if (zncc > cur_max_sum)
							{
								cur_max_sum = zncc;   //update best sum
								best_disp = d;     //update best disparity value
							}
						}
						else
						{
							zncc = CalcZNCC(gray1, gray0, width, height, x, y, d, 1, winsize);
							//if zncc > cur max sum
							if (zncc > cur_max_sum)
							{
								cur_max_sum = zncc;   //update best sum
								best_disp = d;     //update best disparity value
							}
						}
					
					
					}
				}
			}
			//disparity image pixel = best_disp value
			disp_image[4 * y * width + 4 * x + 0] = best_disp;
			disp_image[4 * y * width + 4 * x + 1] = best_disp;
			disp_image[4 * y * width + 4 * x + 2] = best_disp;
			disp_image[4 * y * width + 4 * x + 3] = 255;

		}
	}

	return 0;

}





int main() 
{
	/*
	Steps:
	- read images in
	- scale smaller
	- make grayscale for zncc
	- make disparity maps with zncc
	- cross check
	- occlusion fill
	
	*/

   

	HANDLE myhandle[NUM_THREADS];
	
	/* Timing variables */
	clock_t st, et;
	double took = 0;

	/* Image variables */
	unsigned width, height;
	unsigned char* image0;
	unsigned char* image1;
	unsigned char* scaled0;
	unsigned char* scaled1;
	unsigned char* gray0;
	unsigned char* gray1;

	unsigned char* disp_image0;
	unsigned char* disp_image1;

	unsigned char* crosscheck;
	unsigned char* occlusionfill;
	unsigned char* finale;
	unsigned part = 0;

	//read images
	image0 = ReadImage("im0.png", &width, &height);
	image1 = ReadImage("im1.png", &width, &height);

	/*Downscale images*/

	/* Image0 */

	scaled0 = malloc((width/4) * (height/4) * 4);
	Params params; 
	params.inputimage = image0;
	params.outputimage = scaled0;
	params.width = width;
	params.height = height;

	unsigned threadid;
	Params* pptr = &params; 

	st = clock();
	for (int i = 0; i < NUM_THREADS; i++) 
	{
		myhandle[i] = (HANDLE)_beginthreadex(0, 0, ResizeImage, pptr, 0, &threadid);
	}
	WaitForMultipleObjects(NUM_THREADS, myhandle, 1, INFINITE);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		CloseHandle(myhandle[i]);
	}

	scaled0 = pptr->outputimage;
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Image0 downscaled successfully in %f ms!\n", took);

	/* Image1 */

	scaled1 = malloc((width / 4) * (height / 4) * 4);
	params.inputimage = image1;
	params.outputimage = scaled1;
	pptr = &params;

	st = clock();
	for (int i = 0; i < NUM_THREADS; i++)
	{
		myhandle[i] = (HANDLE)_beginthreadex(0, 0, ResizeImage, pptr, 0, &threadid);
	}
	WaitForMultipleObjects(NUM_THREADS, myhandle, 1, INFINITE);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		CloseHandle(myhandle[i]);
	}

	scaled1 = pptr->outputimage;
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Image1 downscaled successfully in %f ms!\n", took);

	//change these to match new sizes
	width = width / 4;
	height = height / 4;
	printf("Scaled image w x h: %u x %u\n", width, height);

	WriteImage("im0_scaled.png", scaled0, width, height);
	WriteImage("im1_scaled.png", scaled1, width, height);

	/* Grayscale images */

	/* Image 0 */

	gray0 = malloc(width * height * 4);
	params.inputimage = scaled0;
	params.outputimage = gray0;
	params.width = width;
	params.height = height;

	pptr = &params;

	st = clock();
	for (int i = 0; i < NUM_THREADS; i++)
	{
		myhandle[i] = (HANDLE)_beginthreadex(0, 0, GrayScaleImage, pptr, 0, &threadid);
	}
	WaitForMultipleObjects(NUM_THREADS, myhandle, 1, INFINITE);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		CloseHandle(myhandle[i]);
	}

	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Image0 grayscaled successfully in %f ms!\n", took);

	gray0 = pptr->outputimage;
	WriteImage("im0_gray.png", gray0, width, height);

	/* Image 1 */

	gray1 = malloc(width * height * 4);
	params.inputimage = scaled1;
	params.outputimage = gray1;

	pptr = &params;

	st = clock();
	for (int i = 0; i < NUM_THREADS; i++)
	{
		myhandle[i] = (HANDLE)_beginthreadex(0, 0, GrayScaleImage, pptr, 0, &threadid);
	}
	WaitForMultipleObjects(NUM_THREADS, myhandle, 1, INFINITE);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		CloseHandle(myhandle[i]);
	}

	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Image1 grayscaled successfully in %f ms!\n", took);

	gray1 = pptr->outputimage;
	WriteImage("im1_gray.png", gray1, width, height);


	printf("Disparity mapping begins!\n");
	printf("Disparity image 0-1 starts!\n");

	/* Now the images are ready for zncc. */

	/* Image0 to Image1 */

	disp_image0 = malloc(width * height * 4);
	unsigned winsize = 9;

	ParamsZ paramsz;
	paramsz.inputimage = gray0;
	paramsz.inputimage2 = gray1;
	paramsz.outputimage = disp_image0;
	paramsz.width = width;
	paramsz.height = height;
	paramsz.winsize = winsize;
	paramsz.order = 0;

	ParamsZ* pptrz = &paramsz;

	st = clock();
	for (int i = 0; i < NUM_THREADS; i++)
	{
		myhandle[i] = (HANDLE)_beginthreadex(0, 0, DisparityMap, pptrz, 0, &threadid);
	}
	WaitForMultipleObjects(NUM_THREADS, myhandle, 1, INFINITE);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		CloseHandle(myhandle[i]);
	}
	
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Disparity image 0-1 created successfully in %f ms!\n", took);

	disp_image0 = pptrz->outputimage;
	WriteImage("disp01.png", disp_image0, width, height);

	/* Image1 to Image0 */

	printf("Disparity image 1-0 starts!\n");
	
	disp_image1 = malloc(width * height * 4);

	paramsz.order = 1;   //zncc differs based on the order of mapping 

	pptrz = &paramsz;

	st = clock();
	for (int i = 0; i < NUM_THREADS; i++)
	{
		myhandle[i] = (HANDLE)_beginthreadex(0, 0, DisparityMap, pptrz, 0, &threadid);
	}
	WaitForMultipleObjects(NUM_THREADS, myhandle, 1, INFINITE);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		CloseHandle(myhandle[i]);
	}
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Disparity image 1-0 created successfully in %f ms!\n", took);

	disp_image1 = pptrz->outputimage;
	WriteImage("disp10.png", disp_image1, width, height);

	disp_image0 = ReadImage("disp01.png", &width, &height);    //read disp_image0 from memory as disp_imge1 'stole' the pointer

	/* Post processing */

	printf("Cross check begins!\n");

	crosscheck = malloc(width * height * 4);

	paramsz.inputimage = disp_image0;
	paramsz.inputimage2 = disp_image1;
	paramsz.outputimage = crosscheck;

	pptrz = &paramsz;

	st = clock();
	for (int i = 0; i < NUM_THREADS; i++)
	{
		myhandle[i] = (HANDLE)_beginthreadex(0, 0, CrossCheck, pptrz, 0, &threadid);
	}
	WaitForMultipleObjects(NUM_THREADS, myhandle, 1, INFINITE);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		CloseHandle(myhandle[i]);
	}
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Cross check completed successfully in %f ms!\n", took);

	crosscheck = pptrz->outputimage;
	WriteImage("crosscheck.png", crosscheck, width, height);

	/* Occlusion fill */

	printf("Occlusion fill begins!\n");

	occlusionfill = malloc(width * height * 4);

	paramsz.inputimage = crosscheck;
	paramsz.outputimage = occlusionfill;
	paramsz.winsize = 21;                         

	pptrz = &paramsz;

	st = clock();
	for (int i = 0; i < NUM_THREADS; i++)
	{
		myhandle[i] = (HANDLE)_beginthreadex(0, 0, OcclusionFill, pptrz, 0, &threadid);
	}
	WaitForMultipleObjects(NUM_THREADS, myhandle, 1, INFINITE);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		CloseHandle(myhandle[i]);
	}
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Occlusion fill completed successfully in %f ms!\n", took);

	occlusionfill = pptrz->outputimage;
	WriteImage("occlusion.png", occlusionfill, width, height);

	/* Some extra final functionality, value scaling and mean filter. */

	printf("Finalization begins!\n");

	finale = malloc(width * height * 4);

	paramsz.inputimage = occlusionfill;
	paramsz.outputimage = finale;
	paramsz.winsize = 3;

	pptrz = &paramsz;

	st = clock();
	for (int i = 0; i < NUM_THREADS; i++)
	{
		myhandle[i] = (HANDLE)_beginthreadex(0, 0, mean_filter, pptrz, 0, &threadid);
	}
	WaitForMultipleObjects(NUM_THREADS, myhandle, 1, INFINITE);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		CloseHandle(myhandle[i]);
	}
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Finalization completed successfully in %f ms!\n", took);

	finale = pptrz->outputimage;
	WriteImage("final.png", finale, width, height);

	
	/*Free memory*/

	free(image0);
	free(image1);
	free(scaled0);
	free(scaled1);
	free(gray0);
	free(gray1);
	free(disp_image0);
	free(disp_image1);
	free(crosscheck);
	free(occlusionfill);
	free(finale);

	printf("Finish!\n\n");

	return 0;
}