
__kernel void downscale_and_grayscale(__global unsigned char* image, __global unsigned char* result_image, unsigned width, unsigned height)  
{

	unsigned x = get_global_id(0);
	unsigned y = get_global_id(1);
 
	unsigned scale = 4;
	unsigned newwidth = width / scale;
	unsigned char r, g, b, a, gray;

	r = image[scale * 4 * y * width + scale * 4 * x + 0];   /*red*/
	g = image[scale * 4 * y * width + scale * 4 * x + 1];   /*green*/
	b = image[scale * 4 * y * width + scale * 4 * x + 2];   /*blue*/
	a = image[scale * 4 * y * width + scale * 4 * x + 3];   /*alpha*/

	gray = (r*0.30+g*0.59+b*0.11); 

	/* Set same values but every 4th value*/
	result_image[4 * y * newwidth + 4 * x + 0] = gray;
	result_image[4 * y * newwidth + 4 * x + 1] = gray;
	result_image[4 * y * newwidth + 4 * x + 2] = gray;
	result_image[4 * y * newwidth + 4 * x + 3] = a;
}

__kernel void disparitymap(__global unsigned char* gray0, __global unsigned char* gray1, __global unsigned char* disp_image, unsigned width, unsigned height, unsigned winsize, unsigned order) 
{

	unsigned max_disp = 260 / 4;
	double zncc, cur_max_sum, mean0, mean1, sum0, sum1, sum, stdev0, stdev1;
	unsigned best_disp, d, sign, fx, fy;

	unsigned x = get_global_id(0);
	unsigned y = get_global_id(1);

	

	cur_max_sum = 0;
	best_disp = 0;
	for(d = 0; d < max_disp; d += 1)
	{
		if (order == 0) //if mapping gray0 to gray1
		{
			if (d > x)   //if we go over border, use x as max disparity value available 
			{
				sign = 0;

				//avgs for stvdev and zncc
				mean0 = 0;
				sum0 = 0;
				mean1 = 0;
				sum1 = 0;
				for (fy = 0; fy < winsize; fy += 1)  
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0)
						{
							sum0 += gray0[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - ((winsize - 1) / 2))];
							sum1 += gray1[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - x - ((winsize - 1) / 2))];
						}
						else 
						{
							sum0 += gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))];
							sum1 += gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + x - ((winsize - 1) / 2))];
						}
					}
				}
				mean0 = sum0 / (winsize*winsize);
				mean1 = sum1 / (winsize*winsize);

				//stdev for zncc
				sum0 = 0;
				sum1 = 0;
				for (fy = 0; fy < winsize; fy += 1) 
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0)
						{
							sum0 += (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0);
							sum1 += (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - x - ((winsize - 1) / 2))] - mean1) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - x - ((winsize - 1) / 2))] - mean1);
						}
						else 
						{
							sum0 += (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0);
							sum1 += (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + x - ((winsize - 1) / 2))] - mean1) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + x - ((winsize - 1) / 2))] - mean1);
						}
					}
				}
				stdev0 = pow((int)(sum0 / (winsize * winsize)), 0.5);
				stdev1 = pow((int)(sum1 / (winsize * winsize)), 0.5);
				
				//zncc 
				sum = 0;
				for (fy = 0; fy < winsize; fy += 1) 
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0) 
						{
							sum += ((gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - x - ((winsize - 1) / 2))] - mean1));
						}
						else 
						{
							sum += ((gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + x - ((winsize - 1) / 2))] - mean1));
						}
					}
				}
				zncc = sum / (stdev0 * stdev1);

				//if zncc > cur max sum
				if (zncc > cur_max_sum)
				{
					cur_max_sum = zncc;   //update best sum
					best_disp = d;     //update best disparity value
				}
			}
			else
			{
				sign = 0;

				//avgs for stvdev and zncc
				mean0 = 0;
				sum0 = 0;
				mean1 = 0;
				sum1 = 0;
				for (fy = 0; fy < winsize; fy += 1)  
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0)
						{
							sum0 += gray0[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - ((winsize - 1) / 2))];
							sum1 += gray1[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))];
						}
						else 
						{
							sum0 += gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))];
							sum1 += gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))];
						}
					}
				}
				mean0 = sum0 / (winsize*winsize);
				mean1 = sum1 / (winsize*winsize);

				//stdev for zncc
				sum0 = 0;
				sum1 = 0;
				for (fy = 0; fy < winsize; fy += 1) 
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0)
						{
							sum0 += (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0);
							sum1 += (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - mean1) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - mean1);
						}
						else 
						{
							sum0 += (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0);
							sum1 += (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - mean1) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - mean1);
						}
					}
				}
				stdev0 = pow((int)(sum0 / (winsize * winsize)), 0.5);
				stdev1 = pow((int)(sum1 / (winsize * winsize)), 0.5);
				
				//zncc 
				sum = 0;
				for (fy = 0; fy < winsize; fy += 1) 
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0) 
						{
							sum += ((gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - mean1));
						}
						else 
						{
							sum += ((gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - mean1));
						}
					}
				}
				zncc = sum / (stdev0 * stdev1);
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
				//zncc = CalcZNCC(fakegray1, fakegray0, width, height, x, y, (width - x), 1, winsize);
				sign = 1;

				//avgs for stvdev and zncc
				mean0 = 0;
				sum0 = 0;
				mean1 = 0;
				sum1 = 0;
				for (fy = 0; fy < winsize; fy += 1)  
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0)
						{
							sum0 += gray0[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - (width-x) - ((winsize - 1) / 2))];
							sum1 += gray1[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - (width-x) - ((winsize - 1) / 2))];
						}
						else 
						{
							sum0 += gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + (width-x) - ((winsize - 1) / 2))];
							sum1 += gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + (width-x) - ((winsize - 1) / 2))];
						}
					}
				}
				mean0 = sum0 / (winsize*winsize);
				mean1 = sum1 / (winsize*winsize);

				//stdev for zncc
				sum0 = 0;
				sum1 = 0;
				for (fy = 0; fy < winsize; fy += 1) 
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0)
						{
							sum0 += (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - (width-x) - ((winsize - 1) / 2))] - mean0) * (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - (width-x) - ((winsize - 1) / 2))] - mean0);
							sum1 += (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - (width-x) - ((winsize - 1) / 2))] - mean1) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - (width-x) - ((winsize - 1) / 2))] - mean1);
						}
						else 
						{
							sum0 += (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + (width-x) - ((winsize - 1) / 2))] - mean0) * (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + (width-x) - ((winsize - 1) / 2))] - mean0);
							sum1 += (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + (width-x) - ((winsize - 1) / 2))] - mean1) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + (width-x) - ((winsize - 1) / 2))] - mean1);
						}
					}
				}
				stdev0 = pow((int)(sum0 / (winsize * winsize)), 0.5);
				stdev1 = pow((int)(sum1 / (winsize * winsize)), 0.5);
				
				//zncc 
				sum = 0;
				for (fy = 0; fy < winsize; fy += 1) 
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0) 
						{
							sum += ((gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - (width-x) - ((winsize - 1) / 2))] - mean1));
						}
						else 
						{
							sum += ((gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + (width-x) - ((winsize - 1) / 2))] - mean1));
						}
					}
				}
				zncc = sum / (stdev0 * stdev1);

				//if zncc > cur max sum
				if (zncc > cur_max_sum)
				{
					cur_max_sum = zncc;   //update best sum
					best_disp = d;     //update best disparity value
				}
			}
			else
			{
				sign = 1;

				//avgs for stvdev and zncc
				mean0 = 0;
				sum0 = 0;
				mean1 = 0;
				sum1 = 0;
				for (fy = 0; fy < winsize; fy += 1)  
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0)
						{
							sum0 += gray0[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))];
							sum1 += gray1[4 * (y + fy - ((winsize-1)/2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))];
						}
						else 
						{
							sum0 += gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))];
							sum1 += gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))];
						}
					}
				}
				mean0 = sum0 / (winsize*winsize);
				mean1 = sum1 / (winsize*winsize);

				//stdev for zncc
				sum0 = 0;
				sum1 = 0;
				for (fy = 0; fy < winsize; fy += 1) 
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0)
						{
							sum0 += (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - mean0) * (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - mean0);
							sum1 += (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - mean1) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - mean1);
						}
						else 
						{
							sum0 += (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - mean0) * (gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - mean0);
							sum1 += (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - mean1) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - mean1);
						}
					}
				}
				stdev0 = pow((int)(sum0 / (winsize * winsize)), 0.5);
				stdev1 = pow((int)(sum1 / (winsize * winsize)), 0.5);
				
				//zncc 
				sum = 0;
				for (fy = 0; fy < winsize; fy += 1) 
				{
					for (fx = 0; fx < winsize; fx += 1)
					{
						if (sign == 0) 
						{
							sum += ((gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - d - ((winsize - 1) / 2))] - mean1));
						}
						else 
						{
							sum += ((gray0[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx - ((winsize - 1) / 2))] - mean0) * (gray1[4 * (y + fy - ((winsize - 1) / 2)) * width + 4 * (x + fx + d - ((winsize - 1) / 2))] - mean1));
						}
					}
				}
				zncc = sum / (stdev0 * stdev1);
				//if zncc > cur max sum
				if (zncc > cur_max_sum)
				{
					cur_max_sum = zncc;   //update best sum
					best_disp = d;     //update best disparity value
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

__kernel void occlusionfill(__global unsigned char* image, __global unsigned char* result_image, unsigned width, unsigned height, unsigned winsize)
{

	unsigned x = get_global_id(0);
	unsigned y = get_global_id(1);

	unsigned mean, count, fx, fy;

	if (image[4 * y * width + 4 * x] == 0)  //process the black pixels only
	{
		mean = 0;
		count = 1;

		for (fy = 0; fy < winsize; fy += 1)  
		{
			for (fx = 0; fx < winsize; fx += 1)
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

	
	/* Save the mean value as the new gray value and scale the values to range 0-255 */
	result_image[4 * y * width + 4 * x + 0] = 250 / 65 * (mean - 65) + 250;
	result_image[4 * y * width + 4 * x + 1] = 250 / 65 * (mean - 65) + 250;
	result_image[4 * y * width + 4 * x + 2] = 250 / 65 * (mean - 65) + 250;
	result_image[4 * y * width + 4 * x + 3] = 255;

}

__kernel void crosscheck(__global unsigned char* image, __global unsigned char* image2, __global unsigned char* result_image, unsigned width, unsigned height)
{

	unsigned x = get_global_id(0);
	unsigned y = get_global_id(1);

	unsigned diff;
	unsigned treshold = 8;  //8 is default one

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

