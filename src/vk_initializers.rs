use std::slice;

use ash::vk;

pub fn semaphore_create_info<'a>(
    flags: impl Into<Option<vk::SemaphoreCreateFlags>>,
) -> vk::SemaphoreCreateInfo<'a> {
    let flags = flags.into().unwrap_or(vk::SemaphoreCreateFlags::empty());
    vk::SemaphoreCreateInfo::default().flags(flags)
}

pub fn semaphore_submit_info<'a>(
    stage_mask: vk::PipelineStageFlags2,
    semaphore: vk::Semaphore,
) -> vk::SemaphoreSubmitInfo<'a> {
    vk::SemaphoreSubmitInfo::default()
        .semaphore(semaphore)
        .stage_mask(stage_mask)
        .device_index(0)
        .value(1)
}

pub fn fence_create_info<'a>(
    flags: impl Into<Option<vk::FenceCreateFlags>>,
) -> vk::FenceCreateInfo<'a> {
    let flags = flags.into().unwrap_or(vk::FenceCreateFlags::empty());
    vk::FenceCreateInfo::default().flags(flags)
}

pub fn command_pool_create_info<'a>(
    queue_family_index: u32,
    flags: impl Into<Option<vk::CommandPoolCreateFlags>>,
) -> vk::CommandPoolCreateInfo<'a> {
    let flags = flags.into().unwrap_or(vk::CommandPoolCreateFlags::empty());
    vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(flags)
}

pub fn command_buffer_allocate_info<'a>(
    command_pool: vk::CommandPool,
    count: impl Into<Option<u32>>,
) -> vk::CommandBufferAllocateInfo<'a> {
    let count = count.into().unwrap_or(1);
    vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .command_buffer_count(count)
}

pub fn command_buffer_begin_info<'a>(
    flags: impl Into<Option<vk::CommandBufferUsageFlags>>,
) -> vk::CommandBufferBeginInfo<'a> {
    let flags = flags.into().unwrap_or(vk::CommandBufferUsageFlags::empty());
    vk::CommandBufferBeginInfo::default().flags(flags)
}

pub fn command_buffer_submit_info<'a>(cmd: vk::CommandBuffer) -> vk::CommandBufferSubmitInfo<'a> {
    vk::CommandBufferSubmitInfo::default()
        .command_buffer(cmd)
        .device_mask(0)
}

pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
}

pub fn submit_info<'a>(
    cmd_info: &'a vk::CommandBufferSubmitInfo,
    signal_semaphore_info: impl Into<Option<&'a vk::SemaphoreSubmitInfo<'a>>>,
    wait_semaphore_info: impl Into<Option<&'a vk::SemaphoreSubmitInfo<'a>>>,
) -> vk::SubmitInfo2<'a> {
    let signal_semaphores = match signal_semaphore_info.into() {
        Some(info) => slice::from_ref(info),
        None => &[],
    };
    let wait_semaphores = match wait_semaphore_info.into() {
        Some(info) => slice::from_ref(info),
        None => &[],
    };
    vk::SubmitInfo2::default()
        .command_buffer_infos(slice::from_ref(cmd_info))
        .signal_semaphore_infos(signal_semaphores)
        .wait_semaphore_infos(wait_semaphores)
}

pub fn present_info<'a>() -> vk::PresentInfoKHR<'a> {
    vk::PresentInfoKHR::default()
}

pub fn image_create_info<'a>(
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    extent: vk::Extent3D,
) -> vk::ImageCreateInfo<'a> {
    vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .usage(usage)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
}

pub fn image_view_create_info<'a>(
    format: vk::Format,
    image: vk::Image,
    aspect_mask: vk::ImageAspectFlags,
) -> vk::ImageViewCreateInfo<'a> {
    vk::ImageViewCreateInfo::default()
        .format(format)
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .aspect_mask(aspect_mask),
        )
}
