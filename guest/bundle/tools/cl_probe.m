// cl_probe: the OpenCL devices a process sees, and a kernel run on each: c = a + b over 1024
// floats, read back and checked. The GL renderer an accelerator's IOGLBundleName names has compute
// entry points of its own (gldCreateComputeContext and the programs it builds, read from its
// code), which OpenCL may reach, so a change to that personality may change what OpenCL finds and
// runs; this shows it before and after.
//
//   cl_probe        every device of every platform, then the run on each
//
// Prints one line per device ("device N: NAME, TYPE, VENDOR, VERSION, driver DRIVER") and one per
// run ("run N: OK" or "run N: FAIL (what, the OpenCL status)"); exits 0 where every run passed.
// Our own code.
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <stdio.h>
#include <string.h>

static const char *kSource = "__kernel void add(__global const float *a, __global const float *b,\n"
                             "                  __global float *c) {\n"
                             "  size_t i = get_global_id(0);\n"
                             "  c[i] = a[i] + b[i];\n"
                             "}\n";

static const char *type_name(cl_device_type t)
{
	if (t & CL_DEVICE_TYPE_GPU)
		return "GPU";
	if (t & CL_DEVICE_TYPE_CPU)
		return "CPU";
	if (t & CL_DEVICE_TYPE_ACCELERATOR)
		return "accelerator";
	return "other";
}

// The run on `device`: 0 where c holds a + b, else 1 with a line saying what failed.
static int run(int n, cl_device_id device)
{
	enum { kCount = 1024 };
	float a[kCount], b[kCount], c[kCount];
	for (int i = 0; i < kCount; i++) {
		a[i] = (float)i;
		b[i] = (float)(2 * i + 1);
		c[i] = -1.0f;
	}
	cl_int err = CL_SUCCESS;
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (context == NULL) {
		printf("run %d: FAIL (clCreateContext, %d)\n", n, err);
		return 1;
	}
	int bad = 1;
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_mem ma = NULL, mb = NULL, mc = NULL;
	if (queue == NULL) {
		printf("run %d: FAIL (clCreateCommandQueue, %d)\n", n, err);
		goto done;
	}
	program = clCreateProgramWithSource(context, 1, &kSource, NULL, &err);
	if (program == NULL) {
		printf("run %d: FAIL (clCreateProgramWithSource, %d)\n", n, err);
		goto done;
	}
	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		char log[2048] = {0};
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, NULL);
		printf("run %d: FAIL (clBuildProgram, %d: %s)\n", n, err, log);
		goto done;
	}
	kernel = clCreateKernel(program, "add", &err);
	if (kernel == NULL) {
		printf("run %d: FAIL (clCreateKernel, %d)\n", n, err);
		goto done;
	}
	ma = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof a, a, &err);
	mb = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof b, b, &err);
	mc = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof c, NULL, &err);
	if (ma == NULL || mb == NULL || mc == NULL) {
		printf("run %d: FAIL (clCreateBuffer, %d)\n", n, err);
		goto done;
	}
	clSetKernelArg(kernel, 0, sizeof ma, &ma);
	clSetKernelArg(kernel, 1, sizeof mb, &mb);
	clSetKernelArg(kernel, 2, sizeof mc, &mc);
	size_t global = kCount;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("run %d: FAIL (clEnqueueNDRangeKernel, %d)\n", n, err);
		goto done;
	}
	err = clEnqueueReadBuffer(queue, mc, CL_TRUE, 0, sizeof c, c, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("run %d: FAIL (clEnqueueReadBuffer, %d)\n", n, err);
		goto done;
	}
	for (int i = 0; i < kCount; i++) {
		if (c[i] != a[i] + b[i]) {
			printf("run %d: FAIL (c[%d] = %g, want %g)\n", n, i, c[i], a[i] + b[i]);
			goto done;
		}
	}
	printf("run %d: OK\n", n);
	bad = 0;
done:
	if (ma != NULL)
		clReleaseMemObject(ma);
	if (mb != NULL)
		clReleaseMemObject(mb);
	if (mc != NULL)
		clReleaseMemObject(mc);
	if (kernel != NULL)
		clReleaseKernel(kernel);
	if (program != NULL)
		clReleaseProgram(program);
	if (queue != NULL)
		clReleaseCommandQueue(queue);
	clReleaseContext(context);
	return bad;
}

int main(void)
{
	cl_platform_id platforms[8];
	cl_uint platformCount = 0;
	cl_int err = clGetPlatformIDs(8, platforms, &platformCount);
	if (err != CL_SUCCESS) {
		printf("clGetPlatformIDs: %d\n", err);
		return 1;
	}
	cl_device_id devices[32];
	int count = 0;
	for (cl_uint p = 0; p < platformCount; p++) {
		cl_uint n = 0;
		if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 32 - count, devices + count, &n) == CL_SUCCESS)
			count += (int)n;
	}
	for (int i = 0; i < count; i++) {
		char name[256] = {0}, vendor[256] = {0}, version[256] = {0}, driver[256] = {0};
		cl_device_type type = 0;
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof name - 1, name, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof vendor - 1, vendor, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof version - 1, version, NULL);
		clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof driver - 1, driver, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof type, &type, NULL);
		printf("device %d: %s, %s, %s, %s, driver %s\n", i, name, type_name(type), vendor, version, driver);
	}
	int failed = 0;
	for (int i = 0; i < count; i++) {
		fflush(stdout);
		failed += run(i, devices[i]);
	}
	printf("cl_probe: %d devices, %d runs failed\n", count, failed);
	return failed != 0;
}
