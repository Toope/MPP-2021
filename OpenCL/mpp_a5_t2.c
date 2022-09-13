#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x010000) 

//image read and write are done in c

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


int main()
{
	cl_event timer_event = NULL;   //measure input file load, grayscale, filter and saving operations
	cl_ulong start = 0, end = 0;
	cl_double elapsed = 0;

	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem memobj_input0 = NULL;
	cl_mem memobj_input1 = NULL;
	cl_mem memobj_gray0 = NULL;
	cl_mem memobj_gray1 = NULL;
	cl_mem memobj_disp0 = NULL;
	cl_mem memobj_disp1 = NULL;
	cl_mem memobj_crosscheck = NULL;
	cl_mem memobj_finale = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;
	FILE* fp;
	char fileName[] = "mpp_a5_t2.cl";
	
	char* source_str;
	size_t source_size;
	size_t globalWorkSize[2];
	size_t valueSize = 0;
	char* value;


	/* Image variables */
	unsigned width, height;
	unsigned char* image0;
	unsigned char* image1;
	unsigned char* gray0;
	unsigned char* gray1;

	unsigned char* disp_image0;
	unsigned char* disp_image1;

	unsigned char* crosscheck;
	unsigned char* finale;
	unsigned part = 0;

	//NOTE: the image default folder is the Debug folder!


	/* Load the source code containing the kernel*/
	fopen_s(&fp, fileName, "r");

	if (!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)calloc(MAX_SOURCE_SIZE, 1);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* Get Platform and Device Info */
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	printf("Program begins! Showing platform info:\n");
	printf("Platform id:%d, num of platform: %d\n", platform_id, ret_num_platforms);
	printf("Device id:%d, num of device: %d\n", device_id, ret_num_devices);

	// print device name
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valueSize); 
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, value, NULL);
	printf("Device: %s\n", value);
	free(value);

	// print hardware device version
	clGetDeviceInfo(device_id, CL_DEVICE_VERSION, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_VERSION, valueSize, value, NULL);
	printf("Hardware version: %s\n", value);
	free(value);

	// print software driver version
	clGetDeviceInfo(device_id, CL_DRIVER_VERSION, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DRIVER_VERSION, valueSize, value, NULL);
	printf("Software version: %s\n", value);
	free(value);

	// print other wanted info
	cl_device_local_mem_type memtype;  
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, 0, NULL, &valueSize);
	memtype = (cl_device_local_mem_type)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, valueSize, &memtype, NULL);
	printf("LOCAL_MEM_TYPE: %d\n", memtype);

	cl_ulong memsize;
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, 0, NULL, &valueSize);
	memsize = (cl_ulong)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, valueSize, &memsize, NULL);
	printf("LOCAL_MEM_SIZE: %lld\n", memsize);

	cl_uint maxunits;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &valueSize);
	maxunits = (cl_uint)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, valueSize, &maxunits, NULL);
	printf("MAX_COMPUTE_UNITS: %d\n", maxunits);
	 
	cl_uint freq;   
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, 0, NULL, &valueSize);
	freq = (cl_uint)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, valueSize, &freq, NULL);
	printf("MAX_CLOCK_FREQUENCY: %d\n", freq);

	cl_ulong bufsize;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, 0, NULL, &valueSize);
	bufsize = (cl_ulong)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, valueSize, &bufsize, NULL);
	printf("MAX_CONSTANT_BUFFER_SIZE: %d\n", bufsize);

	size_t groupsize;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &valueSize);
	groupsize = (size_t)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, valueSize, &groupsize, NULL);
	printf("MAX_WORK_GROUP_SIZE: %d\n", groupsize);  

	cl_uint dims;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 0, NULL, &valueSize);
	dims = (cl_uint)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, valueSize, &dims, NULL);

	/* Couldn't figure out how to print this. */
	size_t items;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &valueSize);
	items = (size_t)malloc(valueSize);
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, valueSize, items, NULL);
	printf("MAX_WORK_ITEM_SIZES (probably wrong): %d\n\n", items);


	/* Create OpenCL context */ 
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	 
	/* Create Command Queue, that has profiling enabled */
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret); 

	/* Load the image files */

	image0 = ReadImage("im0.png", &width, &height);
	image1 = ReadImage("im1.png", &width, &height);

	//printf("Image w x h: %u x %u\n", width, height);

	/* Downscale and grayscale images */

	gray0 = malloc((width / 4) * (height / 4) * 4);
	gray1 = malloc((width / 4) * (height / 4) * 4);
	disp_image0 = malloc((width / 4) * (height / 4) * 4);
	disp_image1 = malloc((width / 4) * (height / 4) * 4);
	crosscheck = malloc((width / 4) * (height / 4) * 4);
	finale = malloc((width / 4) * (height / 4) * 4);

	/* Create Memory Buffers */
	//create buffers for all image matrices and width and height
	memobj_input0 = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * width * height * sizeof(unsigned char), NULL, &ret);
	memobj_input1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * width * height * sizeof(unsigned char), NULL, &ret);
	memobj_gray0 = clCreateBuffer(context, CL_MEM_READ_WRITE, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), NULL, &ret);
	memobj_gray1 = clCreateBuffer(context, CL_MEM_READ_WRITE, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), NULL, &ret);
	memobj_disp0 = clCreateBuffer(context, CL_MEM_READ_WRITE, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), NULL, &ret);
	memobj_disp1 = clCreateBuffer(context, CL_MEM_READ_WRITE, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), NULL, &ret);
	memobj_crosscheck = clCreateBuffer(context, CL_MEM_READ_WRITE, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), NULL, &ret);
	memobj_finale = clCreateBuffer(context, CL_MEM_READ_WRITE, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), NULL, &ret);
	printf("Memory buffers created!\n");

	/* Copy image matrices to the memory buffer */
	ret = clEnqueueWriteBuffer(command_queue, memobj_input0, CL_TRUE, 0, 4 * width * height * sizeof(unsigned char), image0, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, memobj_input1, CL_TRUE, 0, 4 * width * height * sizeof(unsigned char), image1, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, memobj_gray0, CL_TRUE, 0, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), gray0, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, memobj_gray1, CL_TRUE, 0, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), gray1, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, memobj_disp0, CL_TRUE, 0, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), disp_image0, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, memobj_disp1, CL_TRUE, 0, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), disp_image1, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, memobj_crosscheck, CL_TRUE, 0, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), crosscheck, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, memobj_finale, CL_TRUE, 0, (width / 4) * (height / 4) * 4 * sizeof(unsigned char), finale, 0, NULL, NULL);
	printf("Inputs copied to memory buffers!\n");


	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);

	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		// Allocate memory for the log
		char* log = (char*)malloc(log_size);
		// Get the log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		// Print the log
		printf("%s\n", log);  
	}
	
	printf("Program built!\n");
	 
	/*Params for clEnqueueNDRangeKernel*/
	globalWorkSize[0] = width;
	globalWorkSize[1] = height;

	printf("Down/Grayscale part begin!\n");

	/* Create OpenCL Kernel for grayscale */
	kernel = clCreateKernel(program, "downscale_and_grayscale", &ret);  

	if (ret != CL_SUCCESS)
	{
		printf("Error: Kernel not ok! %d\n", ret);
		exit(1);
	}

	/* IMAGE0 */

	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_input0);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_gray0);
	ret = clSetKernelArg(kernel, 2, sizeof(unsigned), (void*)&width);
	ret = clSetKernelArg(kernel, 3, sizeof(unsigned), (void*)&height);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at setting down/grayscale kernel params! %d\n", ret);
		exit(1);
	}
	else 
	{
		//printf("Down/Grayscale kernel params set!\n");
	}

	/* Execute OpenCL Kernel */

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		globalWorkSize, NULL, 0, NULL, &timer_event);


	clWaitForEvents(1, &timer_event);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	elapsed = (cl_double)(end - start) * (cl_double)(1e-06);

	if (ret != CL_SUCCESS)   
	{
		printf("Error: Failed at down/grayscale NDRangeKernel! %d\n", ret);
		exit(1);
	}
	else 
	{
		printf("Down/Grayscale kernel executed in %f ms!\n", elapsed);
	}

	/* Copy results from the memory buffer*/

	ret = clEnqueueReadBuffer(command_queue, memobj_gray0, CL_TRUE, 0,
		(width / 4) * (height / 4) * 4 * sizeof(unsigned char), gray0, 0, NULL, NULL);


	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at down/grayscale result copy from buffer! %d\n", ret);
		exit(1);
	}
	else 
	{
		//printf("Down/Grayscale result read from buffer!\n");
	}

	/* Save the grayscale image */

	WriteImage("im0_gray.png", gray0, width/4, height/4);

	/* IMAGE1 */

	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_input1);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_gray1);
	ret = clSetKernelArg(kernel, 2, sizeof(unsigned), (void*)&width);
	ret = clSetKernelArg(kernel, 3, sizeof(unsigned), (void*)&height);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at setting down/grayscale kernel params! %d\n", ret);
		exit(1);
	}
	else
	{
		//printf("Down/Grayscale kernel params set!\n");
	}

	/* Execute OpenCL Kernel */

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		globalWorkSize, NULL, 0, NULL, &timer_event);


	clWaitForEvents(1, &timer_event);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	elapsed = (cl_double)(end - start) * (cl_double)(1e-06);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at down/grayscale NDRangeKernel! %d\n", ret);
		exit(1);
	}
	else
	{
		printf("Down/Grayscale kernel executed in %f ms!\n", elapsed);
	}

	/* Copy results from the memory buffer*/

	ret = clEnqueueReadBuffer(command_queue, memobj_gray1, CL_TRUE, 0,
		(width / 4) * (height / 4) * 4 * sizeof(unsigned char), gray1, 0, NULL, NULL);


	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at down/grayscale result copy from buffer! %d\n", ret);
		exit(1);
	}
	else
	{
		//printf("Down/Grayscale result read from buffer!\n");
	}

	/* Save the grayscale image */
	WriteImage("im1_gray.png", gray1, width/4, height/4);


	printf("Disparity mapping begins!\n");
	printf("Disparity image 0-1 starts!\n");

	/* Disp 0-1 */

	unsigned winsize = 9;
	unsigned order = 0;


	/*Params for clEnqueueNDRangeKernel*/
	globalWorkSize[0] = width / 4;
	globalWorkSize[1] = height / 4;

	unsigned smallwidth = width / 4;
	unsigned smallheight = height / 4;

	//printf("Image w x h: %u x %u\n", smallwidth, smallheight);

	/* Create OpenCL Kernel for disparity */
	kernel = clCreateKernel(program, "disparitymap", &ret);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Kernel not ok! %d\n", ret);
		exit(1);
	}
	  
	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_gray0);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_gray1);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memobj_disp0);
	ret = clSetKernelArg(kernel, 3, sizeof(unsigned), (void*)&smallwidth);
	ret = clSetKernelArg(kernel, 4, sizeof(unsigned), (void*)&smallheight);

	ret = clSetKernelArg(kernel, 5, sizeof(unsigned), (void*)&winsize);
	ret = clSetKernelArg(kernel, 6, sizeof(unsigned), (void*)&order);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at setting disparity kernel params! %d\n", ret);
		exit(1);
	}
	else 
	{
		//printf("Disparity kernel params set!\n");
	} 


	/* Execute OpenCL Kernel */

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		globalWorkSize, NULL, 0, NULL, &timer_event);


	clWaitForEvents(1, &timer_event);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	elapsed = (cl_double)(end - start) * (cl_double)(1e-06);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at disparity NDRangeKernel! %d\n", ret);
		exit(1);
	}
	else
	{
		printf("Disparity kernel executed in %f ms!\n", elapsed);
	}  

	/* Copy results from the memory buffer*/

	ret = clEnqueueReadBuffer(command_queue, memobj_disp0, CL_TRUE, 0,
		(width / 4) * (height / 4) * 4 * sizeof(unsigned char), disp_image0, 0, NULL, NULL);


	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at disparity result copy from buffer! %d\n", ret);
		exit(1);
	}
	else
	{
		//printf("Disparity result read from buffer!\n");
	}

	/* Save the disp image */

	WriteImage("disp01.png", disp_image0, width / 4, height / 4);

	/* Disp 1-0 */
	printf("Disparity image 1-0 starts!\n");

	order = 1;

	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_gray1);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_gray0);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memobj_disp1);
	ret = clSetKernelArg(kernel, 3, sizeof(unsigned), (void*)&smallwidth);
	ret = clSetKernelArg(kernel, 4, sizeof(unsigned), (void*)&smallheight);

	ret = clSetKernelArg(kernel, 5, sizeof(unsigned), (void*)&winsize);
	ret = clSetKernelArg(kernel, 6, sizeof(unsigned), (void*)&order);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at setting disparity kernel params! %d\n", ret);
		exit(1);
	}
	else
	{
		//printf("Disparity kernel params set!\n");
	}

	/* Execute OpenCL Kernel */

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		globalWorkSize, NULL, 0, NULL, &timer_event);


	clWaitForEvents(1, &timer_event);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	elapsed = (cl_double)(end - start) * (cl_double)(1e-06);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at disparity NDRangeKernel! %d\n", ret);
		exit(1);
	}
	else
	{
		printf("Disparity kernel executed in %f ms!\n", elapsed);
	}

	/* Copy results from the memory buffer*/

	ret = clEnqueueReadBuffer(command_queue, memobj_disp1, CL_TRUE, 0,
		(width / 4) * (height / 4) * 4 * sizeof(unsigned char), disp_image1, 0, NULL, NULL);


	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at disparity result copy from buffer! %d\n", ret);
		exit(1);
	}
	else
	{
		//printf("Disparity result read from buffer!\n");
	}

	/* Save the disp image */

	WriteImage("disp10.png", disp_image1, width / 4, height / 4);


	printf("Cross check begins!\n");

	/* Create OpenCL Kernel for crosscheck */
	kernel = clCreateKernel(program, "crosscheck", &ret);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Kernel not ok! %d\n", ret);
		exit(1);
	}

	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_disp0);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_disp1);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memobj_crosscheck);
	ret = clSetKernelArg(kernel, 3, sizeof(unsigned), (void*)&smallwidth);
	ret = clSetKernelArg(kernel, 4, sizeof(unsigned), (void*)&smallheight);


	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at setting crosscheck kernel params! %d\n", ret);
		exit(1);
	}
	else
	{
		//printf("Crosscheck kernel params set!\n");
	}


	/* Execute OpenCL Kernel */

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		globalWorkSize, NULL, 0, NULL, &timer_event);


	clWaitForEvents(1, &timer_event);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	elapsed = (cl_double)(end - start) * (cl_double)(1e-06);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at crosscheck NDRangeKernel! %d\n", ret);
		exit(1);
	}
	else
	{
		printf("Crosscheck kernel executed in %f ms!\n", elapsed);
	}

	/* Copy results from the memory buffer*/

	ret = clEnqueueReadBuffer(command_queue, memobj_crosscheck, CL_TRUE, 0,
		(width / 4) * (height / 4) * 4 * sizeof(unsigned char), crosscheck, 0, NULL, NULL);


	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at crosscheck result copy from buffer! %d\n", ret);
		exit(1);
	}
	else
	{
		//printf("Crosscheck result read from buffer!\n");
	}

	/* Save the disp image */

	WriteImage("crosscheck.png", crosscheck, width / 4, height / 4);


	printf("Occlusion fill begins!\n");

	/* Create OpenCL Kernel for crosscheck */
	kernel = clCreateKernel(program, "occlusionfill", &ret);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Kernel not ok! %d\n", ret);
		exit(1);
	}

	winsize = 21;

	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_crosscheck);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_finale);
	ret = clSetKernelArg(kernel, 2, sizeof(unsigned), (void*)&smallwidth);
	ret = clSetKernelArg(kernel, 3, sizeof(unsigned), (void*)&smallheight);
	ret = clSetKernelArg(kernel, 4, sizeof(unsigned), (void*)&winsize);


	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at setting occlusionfill kernel params! %d\n", ret);
		exit(1);
	}
	else
	{
		//printf("Occlusionfill kernel params set!\n");
	}


	/* Execute OpenCL Kernel */

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		globalWorkSize, NULL, 0, NULL, &timer_event);


	clWaitForEvents(1, &timer_event);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(timer_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	elapsed = (cl_double)(end - start) * (cl_double)(1e-06);

	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at occlusionfill NDRangeKernel! %d\n", ret);
		exit(1);
	}
	else
	{
		printf("Occlusionfill kernel executed in %f ms!\n", elapsed);
	}

	/* Copy results from the memory buffer*/

	ret = clEnqueueReadBuffer(command_queue, memobj_finale, CL_TRUE, 0,
		(width / 4) * (height / 4) * 4 * sizeof(unsigned char), finale, 0, NULL, NULL);


	if (ret != CL_SUCCESS)
	{
		printf("Error: Failed at occlusionfill result copy from buffer! %d\n", ret);
		exit(1);
	}
	else
	{
		//printf("Occlusionfill result read from buffer!\n");
	}

	/* Save the disp image */

	WriteImage("finale.png", finale, width / 4, height / 4); 


	/* Finalization */ 

	free(image0); 
	free(image1);
	free(gray0); 
	free(gray1);
	free(disp_image0);
	free(disp_image1);
	free(crosscheck);
	free(finale);
	

	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);

	ret = clReleaseMemObject(memobj_input0);
	ret = clReleaseMemObject(memobj_input1);
	ret = clReleaseMemObject(memobj_gray0);
	ret = clReleaseMemObject(memobj_gray1);
	ret = clReleaseMemObject(memobj_disp0);
	ret = clReleaseMemObject(memobj_disp1);
	ret = clReleaseMemObject(memobj_crosscheck);
	ret = clReleaseMemObject(memobj_finale);



	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(source_str);

	printf("Finished!\n");

	return 0;
}