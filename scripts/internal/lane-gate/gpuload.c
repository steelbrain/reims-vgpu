// gpuload: a fixed GPU load beside a measurement (msi-boot.sh's LOAD): one compute dispatch that
// keeps the GPU busy for DUTY of every PERIOD_MS milliseconds, then nothing for the rest, until
// it is sent SIGTERM or SIGINT. Headless: it opens no window and needs no display.
//   gpuload [-d DUTY] [-p PERIOD_MS] [-g NAME]
//     -d DUTY       the busy share of each period, 0 to 1 (default 0.5)
//     -p PERIOD_MS  the period (default 10)
//     -g NAME       the GPU whose name contains NAME (default: the first discrete GPU)
// The dispatch (gpuload.comp) is sized at the start so that one takes DUTY x PERIOD_MS on the
// GPU it runs on alone, and each second it prints "gpuload: duty=D busy_ms=B" for the last second
// (D the share of the second spent waiting for dispatches), and at its end the whole run's.
// Build (on the host it loads):
//   glslc -mfmt=c gpuload.comp -o gpuload.spv.inc
//   cc -O2 -Wall -Wextra -Werror gpuload.c -lvulkan -o gpuload
#include <vulkan/vulkan.h>

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const uint32_t spv[] =
#include "gpuload.spv.inc"
    ;

static volatile sig_atomic_t stop;
static void on_signal(int s) { (void)s; stop = 1; }

static double now_ms(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define CHECK(x)                                                                                     \
	do {                                                                                             \
		VkResult r_ = (x);                                                                           \
		if (r_ != VK_SUCCESS) {                                                                      \
			fprintf(stderr, "gpuload: %s: %d\n", #x, r_);                                            \
			exit(1);                                                                                 \
		}                                                                                            \
	} while (0)

enum { GROUPS = 1024, OUT_FLOATS = 1024 };

static VkDevice dev;
static VkQueue queue;
static VkCommandBuffer cmd;
static VkFence fence;
static VkPipeline pipeline;
static VkPipelineLayout layout;
static VkDescriptorSet set;

// One dispatch of ITERS a thread, waited for: its milliseconds.
static double run(uint32_t iters)
{
	CHECK(vkResetCommandBuffer(cmd, 0));
	VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	                               .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	CHECK(vkBeginCommandBuffer(cmd, &bi));
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
	vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof iters, &iters);
	vkCmdDispatch(cmd, GROUPS, 1, 1);
	CHECK(vkEndCommandBuffer(cmd));
	VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
	double t0 = now_ms();
	CHECK(vkQueueSubmit(queue, 1, &si, fence));
	CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
	double t = now_ms() - t0;
	CHECK(vkResetFences(dev, 1, &fence));
	return t;
}

int main(int argc, char **argv)
{
	double duty = 0.5, period = 10;
	const char *want = NULL;
	for (int c; (c = getopt(argc, argv, "d:p:g:")) != -1;) {
		if (c == 'd')
			duty = atof(optarg);
		else if (c == 'p')
			period = atof(optarg);
		else if (c == 'g')
			want = optarg;
		else {
			fprintf(stderr, "usage: gpuload [-d DUTY] [-p PERIOD_MS] [-g NAME]\n");
			return 2;
		}
	}
	if (!(duty > 0 && duty < 1) || !(period >= 1)) {
		fprintf(stderr, "gpuload: DUTY must lie between 0 and 1, PERIOD_MS be 1 or more\n");
		return 2;
	}
	signal(SIGTERM, on_signal);
	signal(SIGINT, on_signal);

	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "gpuload",
	                         .apiVersion = VK_API_VERSION_1_1};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	VkInstance inst;
	CHECK(vkCreateInstance(&ici, NULL, &inst));
	uint32_t n = 0;
	CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
	VkPhysicalDevice pds[16];
	n = n > 16 ? 16 : n;
	CHECK(vkEnumeratePhysicalDevices(inst, &n, pds));
	VkPhysicalDevice pd = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties props;
	for (uint32_t i = 0; i < n && pd == VK_NULL_HANDLE; i++) {
		vkGetPhysicalDeviceProperties(pds[i], &props);
		if (want ? strstr(props.deviceName, want) != NULL : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			pd = pds[i];
	}
	if (pd == VK_NULL_HANDLE) {
		fprintf(stderr, "gpuload: no GPU %s%s\n", want ? "named like " : "discrete", want ? want : "");
		return 1;
	}
	uint32_t nf = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, NULL);
	VkQueueFamilyProperties qf[32];
	nf = nf > 32 ? 32 : nf;
	vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, qf);
	uint32_t fam = UINT32_MAX;
	for (uint32_t i = 0; i < nf && fam == UINT32_MAX; i++)
		if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
			fam = i;
	float prio = 1;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = fam,
	                               .queueCount = 1, .pQueuePriorities = &prio};
	VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
	                          .pQueueCreateInfos = &qci};
	CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
	vkGetDeviceQueue(dev, fam, 0, &queue);

	VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = OUT_FLOATS * sizeof(float),
	                          .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
	VkBuffer buf;
	CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(dev, buf, &mr);
	uint32_t type = 0;
	while (!(mr.memoryTypeBits & (1u << type)))
		type++;
	VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
	                            .memoryTypeIndex = type};
	VkDeviceMemory mem;
	CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
	CHECK(vkBindBufferMemory(dev, buf, mem, 0));

	VkDescriptorSetLayoutBinding b = {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                                  .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
	VkDescriptorSetLayoutCreateInfo dli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	                                       .bindingCount = 1, .pBindings = &b};
	VkDescriptorSetLayout dsl;
	CHECK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
	VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(uint32_t)};
	VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
	                                  .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
	CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));
	VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof spv,
	                                .pCode = spv};
	VkShaderModule sm;
	CHECK(vkCreateShaderModule(dev, &smi, NULL, &sm));
	VkComputePipelineCreateInfo cpi = {
	    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
	    .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
	              .module = sm, .pName = "main"},
	    .layout = layout};
	CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipeline));
	VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1};
	VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
	                                  .poolSizeCount = 1, .pPoolSizes = &ps};
	VkDescriptorPool pool;
	CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &pool));
	VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	                                   .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
	CHECK(vkAllocateDescriptorSets(dev, &dai, &set));
	VkDescriptorBufferInfo dbi = {.buffer = buf, .range = VK_WHOLE_SIZE};
	VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .descriptorCount = 1,
	                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi};
	vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
	VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	                                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = fam};
	VkCommandPool cp;
	CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
	VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
	                                   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
	CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
	VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	CHECK(vkCreateFence(dev, &fci, NULL, &fence));

	// The dispatch's size: ITERS scaled until one dispatch takes the period's busy share.
	double busy = duty * period;
	uint32_t iters = 1024;
	run(iters);
	for (int i = 0; i < 8; i++) {
		double t = run(iters);
		double next = iters * busy / (t > 0.01 ? t : 0.01);
		iters = next < 16 ? 16 : next > 1e8 ? 100000000u : (uint32_t)next;
	}
	printf("gpuload: %s, duty %.2f of %.0f ms, %u iterations a thread, %d threads\n", props.deviceName, duty, period,
	       iters, GROUPS * 256);
	fflush(stdout);

	double start = now_ms(), sec = start, sec_busy = 0, all_busy = 0;
	while (!stop) {
		double t0 = now_ms();
		double t = run(iters);
		sec_busy += t;
		all_busy += t;
		double end = t0 + period;
		double left = end - now_ms();
		if (left > 0)
			usleep((useconds_t)(left * 1e3));
		double tn = now_ms();
		if (tn - sec >= 1000) {
			printf("gpuload: duty=%.3f busy_ms=%.1f\n", sec_busy / (tn - sec), sec_busy);
			fflush(stdout);
			sec = tn;
			sec_busy = 0;
		}
	}
	double total = now_ms() - start;
	printf("gpuload: end duty=%.3f seconds=%.1f\n", total > 0 ? all_busy / total : 0, total / 1e3);
	vkDeviceWaitIdle(dev);
	vkDestroyDevice(dev, NULL);
	vkDestroyInstance(inst, NULL);
	return 0;
}
