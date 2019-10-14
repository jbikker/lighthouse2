/* vulkan_shader.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

namespace lh2core
{

class VulkanShader
{
  public:
	VulkanShader() = default;
	VulkanShader( const VulkanDevice &device, const std::string_view &fileName, const std::vector<std::pair<std::string, std::string>> &definitions = {} );
	~VulkanShader();

	void Cleanup();
	vk::PipelineShaderStageCreateInfo GetShaderStage( vk::ShaderStageFlagBits stage ) const;

	operator vk::ShaderModule() const { return m_Module; }

	std::string PreprocessShader( const std::string_view &fileName, const std::string &source,
								  shaderc_shader_kind shaderKind = shaderc_glsl_infer_from_source );
	std::string CompileToAssembly( const std::string_view &fileName, const std::string &source,
								   shaderc_shader_kind shaderKind = shaderc_glsl_infer_from_source );
	std::vector<uint32_t> CompileFile( const std::string_view &fileName, const std::string &source,
									   shaderc_shader_kind shaderKind = shaderc_glsl_infer_from_source );

	static std::string BaseFolder;
	static std::string BSDFFolder;

  private:
	// Helper classes taken from glslc: https://github.com/google/shaderc
	class FileFinder
	{
	  public:
		std::string FindReadableFilepath( const std::string &filename ) const;
		std::string FindRelativeReadableFilepath( const std::string &requesting_file, const std::string &filename ) const;
		std::vector<std::string> &search_path() { return search_path_; }

	  private:
		std::vector<std::string> search_path_;
	};
	class FileIncluder : public shaderc::CompileOptions::IncluderInterface
	{
	  public:
		explicit FileIncluder( const FileFinder *file_finder ) : file_finder_( *file_finder ) {}

		~FileIncluder() override;
		shaderc_include_result *GetInclude( const char *requested_source,
											shaderc_include_type type,
											const char *requesting_source,
											size_t include_depth ) override;
		void ReleaseInclude( shaderc_include_result *include_result ) override;
		const std::unordered_set<std::string> &file_path_trace() const { return included_files_; }

	  private:
		const FileFinder &file_finder_;
		struct FileInfo
		{
			const std::string full_path;
			std::vector<char> contents;
		};
		std::unordered_set<std::string> included_files_;
	};

	VulkanDevice m_Device;
	shaderc::Compiler m_Compiler;
	shaderc::CompileOptions m_CompileOptions;
	FileFinder m_Finder{};
	vk::ShaderModule m_Module = nullptr;

	static std::vector<char> ReadFile( const std::string_view &fileName );
	static std::string ReadTextFile( const std::string_view &fileName );
};

} // namespace lh2core