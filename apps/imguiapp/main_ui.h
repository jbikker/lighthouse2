//  +-----------------------------------------------------------------------------+
//  |  HandleMaterialChange                                                       |
//  |  Update a scene material based on AntTweakBar.                        LH2'19|
//  +-----------------------------------------------------------------------------+
void UpdateUI()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGui::Begin( "Render statistics", 0 );
	coreStats = renderer->GetCoreStats();
	SystemStats systemStats = renderer->GetSystemStats();
	ImGui::Text( "Frame time:   %6.2fms", coreStats.renderTime * 1000 );
	ImGui::Text( "Scene update: %6.2fms", systemStats.sceneUpdateTime * 1000 );
	ImGui::Text( "Primary rays: %6.2fms", coreStats.traceTime0 * 1000 );
	ImGui::Text( "Secondary:    %6.2fms", coreStats.traceTime1 * 1000 );
	ImGui::Text( "Deep rays:    %6.2fms", coreStats.traceTimeX * 1000 );
	ImGui::Text( "Shadow rays:  %6.2fms", coreStats.shadowTraceTime * 1000 );
	ImGui::Text( "Shading time: %6.2fms", coreStats.shadeTime * 1000 );
	ImGui::Text( "Filter time:  %6.2fms", coreStats.filterTime * 1000 );
	ImGui::Text( "# primary:    %6ik (%6.1fM/s)", coreStats.primaryRayCount / 1000, coreStats.primaryRayCount / (max( 1.0f, coreStats.traceTime0 * 1000000 )) );
	ImGui::Text( "# secondary:  %6ik (%6.1fM/s)", coreStats.bounce1RayCount / 1000, coreStats.bounce1RayCount / (max( 1.0f, coreStats.traceTime1 * 1000000 )) );
	ImGui::Text( "# deep rays:  %6ik (%6.1fM/s)", coreStats.deepRayCount / 1000, coreStats.deepRayCount / (max( 1.0f, coreStats.traceTimeX * 1000000 )) );
	ImGui::Text( "# shadw rays: %6ik (%6.1fM/s)", coreStats.totalShadowRays / 1000, coreStats.totalShadowRays / (max( 1.0f, coreStats.shadowTraceTime * 1000000 )) );
	ImGui::End();
	ImGui::Begin( "Camera parameters", 0 );
	float3 camPos = renderer->GetCamera()->position;
	float3 camDir = renderer->GetCamera()->direction;
	ImGui::Text( "position: %5.2f, %5.2f, %5.2f", camPos.x, camPos.y, camPos.z );
	ImGui::Text( "viewdir:  %5.2f, %5.2f, %5.2f", camDir.x, camDir.y, camDir.z );
	ImGui::SliderFloat( "FOV", &renderer->GetCamera()->FOV, 10, 90 );
	ImGui::SliderFloat( "aperture", &renderer->GetCamera()->aperture, 0, 0.025f );
	ImGui::SliderFloat( "distortion", &renderer->GetCamera()->distortion, 0, 0.5f );
	ImGui::Combo( "tonemap", &renderer->GetCamera()->tonemapper, "clamp\0reinhard\0reinhard ext\0reinhard lum\0reinhard jodie\0uncharted2\0\0" );
	ImGui::SliderFloat( "brightness", &renderer->GetCamera()->brightness, 0, 0.5f );
	ImGui::SliderFloat( "contrast", &renderer->GetCamera()->contrast, 0, 0.5f );
	ImGui::SliderFloat( "gamma", &renderer->GetCamera()->gamma, 1, 2.5f );
	ImGui::End();
	ImGui::Begin( "Material parameters", 0 );
	ImGui::Text( "name:    %s", currentMaterial.name.c_str() );
	ImGui::ColorEdit3( "color", (float*)&currentMaterial.color() );
	ImGui::ColorEdit3( "absorption", (float*)&currentMaterial.absorption() );
	ImGui::SliderFloat( "metallic", &currentMaterial.metallic(), 0, 1 );
	ImGui::SliderFloat( "subsurface", &currentMaterial.subsurface(), 0, 1 );
	ImGui::SliderFloat( "specular", &currentMaterial.specular(), 0, 1 );
	ImGui::SliderFloat( "roughness", &currentMaterial.roughness(), 0, 1 );
	ImGui::SliderFloat( "specularTint", &currentMaterial.specularTint(), 0, 1 );
	ImGui::SliderFloat( "anisotropic", &currentMaterial.anisotropic(), 0, 1 );
	ImGui::SliderFloat( "sheen", &currentMaterial.sheen(), 0, 1 );
	ImGui::SliderFloat( "sheenTint", &currentMaterial.sheenTint(), 0, 1 );
	ImGui::SliderFloat( "clearcoat", &currentMaterial.clearcoat(), 0, 1 );
	ImGui::SliderFloat( "clearcoatGloss", &currentMaterial.clearcoatGloss(), 0, 1 );
	ImGui::SliderFloat( "transmission", &currentMaterial.transmission(), 0, 1 );
	ImGui::SliderFloat( "eta (1/ior)", &currentMaterial.eta(), 0.25f, 1.0f );
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
}

// EOF