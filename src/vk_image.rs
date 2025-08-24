use std::slice;

use ash::vk;

use crate::vk_initializers as vkinit;

pub fn transition_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    current_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let mut image_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
        .old_layout(current_layout)
        .new_layout(new_layout);

    let aspect_mask = match new_layout {
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => vk::ImageAspectFlags::DEPTH,
        _ => vk::ImageAspectFlags::COLOR,
    };
    image_barrier.subresource_range = vkinit::image_subresource_range(aspect_mask);
    image_barrier.image = image;

    let dep_info =
        vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_barrier));

    unsafe { device.cmd_pipeline_barrier2(cmd, &dep_info) };
}
