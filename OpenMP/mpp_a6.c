/*
Course work for 'Multiprocessor Programming' - spring 2021
Task: Stereo disparity OpenMP implementation
By: Tiia Leinonen
*/

#include "lodepng.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <math.h>


#define NUM_THREADS 4    //4 threads in the beginning for testing

/* Helper functions for zncc */

float calc_average(unsigned char* image, unsigned width, unsigned height, unsigned x, unsigned y, unsigned d, int sign, unsigned winsize) 
{
	float mean, sum;
	unsigned fx, fy;

	mean = 0;
	sum = 0;
	for (fy = 0; fy < winsize; fy += 1)   //loop basically from -2 to 2
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
	for (fy = 0; fy < winsize; fy += 1)   //loop basically from -2 to 2
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

unsigned char* mean_filter(unsigned char* image, signed width, signed height, signed winsize)   //takes in an image, returns it filtered
{
	unsigned char* result_image = malloc(width * height * 4);  
	signed x = 0, y = 0, fy = 0, fx = 0;
	unsigned char mean = 0;
	unsigned sum = 0;


	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel for private(x,sum,mean,fx,fy)
	for (y = 0; y < height; y += 1)    //jump because one rgba value takes 4 numbers
	{
		for (x = 0; x < width; x += 1)
		{

			if (x < ((winsize - 1) / 2) || y < ((winsize - 1) / 2) || x > width - ((winsize - 1) / 2) - 1 || y > height - ((winsize - 1) / 2) - 1)
			{
				mean = image[4 * y * width + 4 * x];
			}
			else
			{
				mean = 0;
				sum = 0;
				for (fy = 0; fy < winsize; fy += 1)   //loop basically from -2 to 2
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

	return result_image;
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

unsigned char* ResizeImage(unsigned char* image, signed width, signed height)
{ /* Resizes the image down by scale 4 */

	signed x = 0, y = 0;
	signed scale = 4;;

	signed newwidth = width / scale;
	signed newheight = height / scale;
	unsigned char* result_image = malloc(newwidth * newheight * 4);  
	unsigned char r = 0, g = 0, b = 0, a = 0;

	/* Get RGBA components */
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel for private(r,g,b,a,x)
	for (y = 0; y < newheight; y += 1)
	{
		//printf("%d\n", omp_get_thread_num());
		for (x = 0; x < newwidth; x += 1)
		{

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
	

	return result_image;
}

unsigned char* GrayScaleImage(unsigned char* image, signed width, signed height)
{

	unsigned char* result_image = malloc(width * height * 4); 
	signed x = 0, y = 0;
	unsigned char r = 0, g = 0, b = 0, a = 0, gray = 0;

	/* Get RGBA components */
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel for private(r,g,b,a,gray,x)
	for (y = 0; y < height; y += 1) 
	{
		for (x = 0; x < width; x += 1)
		{
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

	return result_image;
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

unsigned char* CrossCheck(unsigned char* image, unsigned char* image2, signed width, signed height)
{
	unsigned char* result_image = malloc(width * height * 4); 
	signed treshold = 8;  //8 is default one
	signed diff = 0, x = 0, y = 0;

	//normalize values newvalue= (max'-min')/(max-min)*(value-max)+max'
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel for private(x, diff)
	for (y = 0; y < height; y += 1)
	{
		for (x = 0; x < width; x += 1)
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

	return result_image;
}

unsigned char* OcclusionFill(unsigned char* image, unsigned width, unsigned height, unsigned winsize)
{
	unsigned char* result_image = malloc(width * height * 4); 
	size_t jump = 1;
	unsigned mean, treshold, count;

	for (unsigned y = 0; y + jump - 1 < height; y += jump) 
	{
		for (unsigned x = 0; x + jump - 1 < width; x += jump)
		{
			if (x < ((winsize - 1) / 2) || y < ((winsize - 1) / 2) || x > width - ((winsize - 1) / 2) -1 || y > height - ((winsize - 1) / 2) -1)
			{
				mean = image[4 * y * width + 4 * x];
			}
			else {
				if (image[4 * y * width + 4 * x] == 0)  //process the black pixels only
				{
					treshold = 50;
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


	return result_image;

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

unsigned char* DisparityMap(unsigned char* gray0, unsigned char* gray1, signed width, signed height, signed winsize, signed order)
{

	unsigned char* disp_image = malloc(width * height * 4);
	signed x = 0, y = 0;
	signed max_disp = 260 / 4;
	float zncc = 0, cur_max_sum = 0;
	signed best_disp = 0;

	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel for private(x,best_disp,cur_max_sum,zncc)
	for (y = 0; y < height; y += 1)
	{
		for (x = 0; x < width; x += 1)
		{
			if (x < ((winsize - 1) / 2) || y < ((winsize - 1) / 2) || x > width - ((winsize - 1) / 2) - 1 || y > height - ((winsize - 1) / 2) - 1)
			{
				best_disp = 0;
			}
			else {
				cur_max_sum = 0;
				best_disp = 0;
				for (signed d = 0; d < max_disp; d++)
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
						if (d > (signed)(width - x))   //if we go over border, use width-x as max disparity value available 
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

	return disp_image;

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
	- finalizations 
	*/

	/* Timing variables */
	clock_t st, et, tst, tet;
	double took = 0;

	tst = clock();   //total execution time start

	/* Image variables */
	unsigned width, height;
	unsigned char* image0;
	unsigned char* image1;
	unsigned char* scaled0;
	unsigned char* scaled1;
	unsigned char* gray0;
	unsigned char* gray1;

	unsigned char* crosscheck;
	unsigned char* occlusionfill;
	unsigned char* finale;


	//NOTE: the image default folder is the Debug folder!

	image0 = ReadImage("im0.png", &width, &height);
	//easy way to get image dimensions out
	//printf("Image0 w x h: %u x %u\n", width, height);

	image1 = ReadImage("im1.png", &width, &height);
	//easy way to get image dimensions out
	//printf("Image1 w x h: %u x %u\n", width, height);

	st = clock();
	scaled0 = ResizeImage(image0, width, height);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Image0 downscaled successfully in %f ms!\n", took);

	st = clock();
	scaled1 = ResizeImage(image1, width, height);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Image1 downscaled successfully in %f ms!\n", took);

	//change these to match new sizes
	width = width / 4;
	height = height / 4;
	//printf("Scaled image w x h: %u x %u\n", width, height);

	WriteImage("im0_scaled.png", scaled0, width, height);
	WriteImage("im1_scaled.png", scaled1, width, height);

	st = clock();
	gray0 = GrayScaleImage(scaled0, width, height);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Image0 grayscaled successfully in %f ms!\n", took);

	st = clock();
	gray1 = GrayScaleImage(scaled1, width, height);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Image1 grayscaled successfully in %f ms!\n", took);


	WriteImage("im0_gray.png", gray0, width, height);
	WriteImage("im1_gray.png", gray1, width, height);

	printf("Disparity mapping begins!\n");
	printf("Disparity image 0-1 starts!\n");

	/* Now the images are ready for zncc. */

	unsigned winsize = 9;
	unsigned char* disp_image0 = malloc(width * height * 4);
	unsigned char* disp_image1 = malloc(width * height * 4); 

	st = clock();
	disp_image0 = DisparityMap(gray0, gray1, width, height, winsize, 0);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Disparity image 0-1 created successfully in %f ms!\n", took);

	WriteImage("disp01.png", disp_image0, width, height);

	printf("Disparity image 1-0 starts!\n");
	
	//now map the images the other way around

	st = clock();
	disp_image1 = DisparityMap(gray0, gray1, width, height, winsize, 1);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Disparity image 1-0 created successfully in %f ms!\n", took);

	WriteImage("disp10.png", disp_image1, width, height);

	/* Post processing */

	printf("Cross check begins!\n");
	st = clock();
	crosscheck = CrossCheck(disp_image0, disp_image1, width, height);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Cross check completed successfully in %f ms!\n", took);

	WriteImage("crosscheck.png", crosscheck, width, height);

	printf("Occlusion fill begins!\n");
	st = clock();
	occlusionfill = OcclusionFill(crosscheck, width, height, 21);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Occlusion fill completed successfully in %f ms!\n", took);

	WriteImage("occlusion.png", occlusionfill, width, height);

	printf("Finalization begins!\n");
	st = clock();
	finale = mean_filter(occlusionfill, width, height, 3);
	et = clock();
	took = ((double)et - st) / CLOCKS_PER_SEC * 1000;
	printf("Finalization completed successfully in %f ms!\n", took);

	WriteImage("final.png", finale, width, height);

	
	/*Free memory*/

	free(image0);
	free(image1);
	free(scaled0);
	free(scaled1);
	free(gray0);
	free(gray1);
	free(disp_image0);
	free(crosscheck);
	free(occlusionfill);
	free(finale);

	tet = clock();
	took = ((double)tet - tst) / CLOCKS_PER_SEC * 1000;

	printf("Finish in %f seconds!\n\n", took/1000);

	return 0;
}