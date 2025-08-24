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

pub fn copy_image_to_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    source: vk::Image,
    destination: vk::Image,
    src_size: vk::Extent2D,
    dst_size: vk::Extent2D,
) {
    let image_blit = vk::ImageBlit2::default()
        .src_offsets([
            vk::Offset3D::default(),
            vk::Offset3D::default()
                .x(src_size.width as i32)
                .y(src_size.height as i32)
                .z(1),
        ])
        .dst_offsets([
            vk::Offset3D::default(),
            vk::Offset3D::default()
                .x(dst_size.width as i32)
                .y(dst_size.height as i32)
                .z(1),
        ])
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0),
        )
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0),
        );
    let blit_info = vk::BlitImageInfo2::default()
        .src_image(source)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(destination)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(slice::from_ref(&image_blit));

    unsafe { device.cmd_blit_image2(cmd, &blit_info) };
}
