use std::fs;
use std::path::Path;

use ash::vk;

pub fn load_shader_module<P: AsRef<Path>>(
    device: &ash::Device,
    path: P,
) -> Result<vk::ShaderModule, anyhow::Error> {
    let data = fs::read(path)?;

    let code: Vec<_> = data
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

    Ok(unsafe { device.create_shader_module(&create_info, None)? })
}
