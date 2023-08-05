#define LVK_BUILDING_SHADER_COMPILE

#include "ShaderCompile.h"

#include <glslang/Include/glslang_c_interface.h>

#include <cassert>
#include <lvk/vulkan/VulkanUtils.h>
#include <ldrutils/lutils/ScopeExit.h>

VkPhysicalDevice last_device_;
glslang_resource_t resource_template_;

void lvk::shader::initialize() {
	glslang_initialize_process();
}

void lvk::shader::shutdown() {
	glslang_finalize_process();
}

static glslang_stage_t getGLSLangShaderStage(VkShaderStageFlagBits stage) {
	switch (stage) {
		case VK_SHADER_STAGE_VERTEX_BIT:
			return GLSLANG_STAGE_VERTEX;
		case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
			return GLSLANG_STAGE_TESSCONTROL;
		case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
			return GLSLANG_STAGE_TESSEVALUATION;
		case VK_SHADER_STAGE_GEOMETRY_BIT:
			return GLSLANG_STAGE_GEOMETRY;
		case VK_SHADER_STAGE_FRAGMENT_BIT:
			return GLSLANG_STAGE_FRAGMENT;
		case VK_SHADER_STAGE_COMPUTE_BIT:
			return GLSLANG_STAGE_COMPUTE;
		default:
			return GLSLANG_STAGE_COUNT;
	};
	// This will hopefully silence MSVC.
	return GLSLANG_STAGE_COUNT;
}

void logShaderError(const char *szProblem, glslang_shader_t *shader) {
	auto appendIfNonNull = [&](const char *szLog) {
		if (szLog && *szLog)
			LLOGW("%s\n", szLog);
	};

	appendIfNonNull(szProblem);
	appendIfNonNull(glslang_shader_get_info_log(shader));
	appendIfNonNull(glslang_shader_get_info_debug_log(shader));
}

lvk::Result lvk::shader::compile(VkDevice device,
								 VkShaderStageFlagBits stage,
								 const char *code,
								 VkShaderModule *outShaderModule) {
	IGL_PROFILER_FUNCTION();

	if (!outShaderModule) {
		return Result(Result::Code::ArgumentOutOfRange, "outShaderModule is NULL");
	}

	glslang_stage_t apiStage = getGLSLangShaderStage(stage);
	if (apiStage == GLSLANG_STAGE_COUNT) {
		return Result(Result::Code::RuntimeError, "Couldn't cast VkShaderStageFlagBits to a GLSL shader stage");
	}

	const glslang_input_t input = {
			.language = GLSLANG_SOURCE_GLSL,
			.stage = apiStage,
			.client = GLSLANG_CLIENT_VULKAN,
			.client_version = GLSLANG_TARGET_VULKAN_1_3,
			.target_language = GLSLANG_TARGET_SPV,
			.target_language_version = GLSLANG_TARGET_SPV_1_6,
			.code = code,
			.default_version = 100,
			.default_profile = GLSLANG_NO_PROFILE,
			.force_default_version_and_profile = false,
			.forward_compatible = false,
			.messages = GLSLANG_MSG_DEFAULT_BIT,
			.resource = &resource_template_,
	};

	glslang_shader_t *shader = glslang_shader_create(&input);
	SCOPE_EXIT {
				   glslang_shader_delete(shader);
			   };

	if (!glslang_shader_preprocess(shader, &input)) {
		logShaderError("Shader preprocessing failed:\n", shader);
		return Result(Result::Code::RuntimeError, "glslang_shader_preprocess() failed");
	}

	if (!glslang_shader_parse(shader, &input)) {
		logShaderError("Shader parsing failed:\n", shader);
		return Result(Result::Code::RuntimeError, "glslang_shader_parse() failed");
	}

	glslang_program_t *program = glslang_program_create();
	glslang_program_add_shader(program, shader);

	SCOPE_EXIT {
				   glslang_program_delete(program);
			   };

	if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
		logShaderError("Shader linking failed:\n", shader);
		return Result(Result::Code::RuntimeError, "glslang_program_link() failed");
	}

	glslang_spv_options_t options = {
			.generate_debug_info = true,
			.strip_debug_info = false,
			.disable_optimizer = false,
			.optimize_size = true,
			.disassemble = false,
			.validate = true,
			.emit_nonsemantic_shader_debug_info = false,
			.emit_nonsemantic_shader_debug_source = false,
	};

	glslang_program_SPIRV_generate_with_options(program, input.stage, &options);

	if (glslang_program_SPIRV_get_messages(program)) {
		LLOGW("%s\n", glslang_program_SPIRV_get_messages(program));
	}

	const VkShaderModuleCreateInfo ci = {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = glslang_program_SPIRV_get_size(program) * sizeof(uint32_t),
			.pCode = glslang_program_SPIRV_get_ptr(program),
	};
	VK_ASSERT_RETURN(vkCreateShaderModule(device, &ci, nullptr, outShaderModule));

	return Result();
}

glslang_resource_t getGlslangResource(const VkPhysicalDeviceLimits &limits) {
	const glslang_resource_t resource = {
			.max_lights = 32,
			.max_clip_planes = 6,
			.max_texture_units = 32,
			.max_texture_coords = 32,
			.max_vertex_attribs = (int) limits.maxVertexInputAttributes,
			.max_vertex_uniform_components = 4096,
			.max_varying_floats = 64,
			.max_vertex_texture_image_units = 32,
			.max_combined_texture_image_units = 80,
			.max_texture_image_units = 32,
			.max_fragment_uniform_components = 4096,
			.max_draw_buffers = 32,
			.max_vertex_uniform_vectors = 128,
			.max_varying_vectors = 8,
			.max_fragment_uniform_vectors = 16,
			.max_vertex_output_vectors = 16,
			.max_fragment_input_vectors = 15,
			.min_program_texel_offset = -8,
			.max_program_texel_offset = 7,
			.max_clip_distances = (int) limits.maxClipDistances,
			.max_compute_work_group_count_x = (int) limits.maxComputeWorkGroupCount[0],
			.max_compute_work_group_count_y = (int) limits.maxComputeWorkGroupCount[1],
			.max_compute_work_group_count_z = (int) limits.maxComputeWorkGroupCount[2],
			.max_compute_work_group_size_x = (int) limits.maxComputeWorkGroupSize[0],
			.max_compute_work_group_size_y = (int) limits.maxComputeWorkGroupSize[1],
			.max_compute_work_group_size_z = (int) limits.maxComputeWorkGroupSize[2],
			.max_compute_uniform_components = 1024,
			.max_compute_texture_image_units = 16,
			.max_compute_image_uniforms = 8,
			.max_compute_atomic_counters = 8,
			.max_compute_atomic_counter_buffers = 1,
			.max_varying_components = 60,
			.max_vertex_output_components = (int) limits.maxVertexOutputComponents,
			.max_geometry_input_components = (int) limits.maxGeometryInputComponents,
			.max_geometry_output_components = (int) limits.maxGeometryOutputComponents,
			.max_fragment_input_components = (int) limits.maxFragmentInputComponents,
			.max_image_units = 8,
			.max_combined_image_units_and_fragment_outputs = 8,
			.max_combined_shader_output_resources = 8,
			.max_image_samples = 0,
			.max_vertex_image_uniforms = 0,
			.max_tess_control_image_uniforms = 0,
			.max_tess_evaluation_image_uniforms = 0,
			.max_geometry_image_uniforms = 0,
			.max_fragment_image_uniforms = 8,
			.max_combined_image_uniforms = 8,
			.max_geometry_texture_image_units = 16,
			.max_geometry_output_vertices = (int) limits.maxGeometryOutputVertices,
			.max_geometry_total_output_components = (int) limits.maxGeometryTotalOutputComponents,
			.max_geometry_uniform_components = 1024,
			.max_geometry_varying_components = 64,
			.max_tess_control_input_components =
			(int) limits.maxTessellationControlPerVertexInputComponents,
			.max_tess_control_output_components =
			(int) limits.maxTessellationControlPerVertexOutputComponents,
			.max_tess_control_texture_image_units = 16,
			.max_tess_control_uniform_components = 1024,
			.max_tess_control_total_output_components = 4096,
			.max_tess_evaluation_input_components = (int) limits.maxTessellationEvaluationInputComponents,
			.max_tess_evaluation_output_components =
			(int) limits.maxTessellationEvaluationOutputComponents,
			.max_tess_evaluation_texture_image_units = 16,
			.max_tess_evaluation_uniform_components = 1024,
			.max_tess_patch_components = 120,
			.max_patch_vertices = 32,
			.max_tess_gen_level = 64,
			.max_viewports = (int) limits.maxViewports,
			.max_vertex_atomic_counters = 0,
			.max_tess_control_atomic_counters = 0,
			.max_tess_evaluation_atomic_counters = 0,
			.max_geometry_atomic_counters = 0,
			.max_fragment_atomic_counters = 8,
			.max_combined_atomic_counters = 8,
			.max_atomic_counter_bindings = 1,
			.max_vertex_atomic_counter_buffers = 0,
			.max_tess_control_atomic_counter_buffers = 0,
			.max_tess_evaluation_atomic_counter_buffers = 0,
			.max_geometry_atomic_counter_buffers = 0,
			.max_fragment_atomic_counter_buffers = 1,
			.max_combined_atomic_counter_buffers = 1,
			.max_atomic_counter_buffer_size = 16384,
			.max_transform_feedback_buffers = 4,
			.max_transform_feedback_interleaved_components = 64,
			.max_cull_distances = (int) limits.maxCullDistances,
			.max_combined_clip_and_cull_distances = (int) limits.maxCombinedClipAndCullDistances,
			.max_samples = 4,
			.max_mesh_output_vertices_nv = 256,
			.max_mesh_output_primitives_nv = 512,
			.max_mesh_work_group_size_x_nv = 32,
			.max_mesh_work_group_size_y_nv = 1,
			.max_mesh_work_group_size_z_nv = 1,
			.max_task_work_group_size_x_nv = 32,
			.max_task_work_group_size_y_nv = 1,
			.max_task_work_group_size_z_nv = 1,
			.max_mesh_view_count_nv = 4,
			.maxDualSourceDrawBuffersEXT = 1,
			.limits = {
					.non_inductive_for_loops = true,
					.while_loops = true,
					.do_while_loops = true,
					.general_uniform_indexing = true,
					.general_attribute_matrix_vector_indexing = true,
					.general_varying_indexing = true,
					.general_sampler_indexing = true,
					.general_variable_indexing = true,
					.general_constant_matrix_vector_indexing = true,
			}};

	return resource;
}

VkPhysicalDevice lvk::shader::getTargetPhysicalDevice() {
	return last_device_;
}

void lvk::shader::loadTargetPhysicalDeviceLimits(VkPhysicalDevice device, const VkPhysicalDeviceLimits &limits) {
	last_device_ = device;
	resource_template_ = getGlslangResource(limits);
}