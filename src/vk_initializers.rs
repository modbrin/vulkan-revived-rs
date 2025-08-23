use ash::vk;

pub fn semaphore_create_info<'a>(
    flags: impl Into<Option<vk::SemaphoreCreateFlags>>,
) -> vk::SemaphoreCreateInfo<'a> {
    let flags = flags.into().unwrap_or(vk::SemaphoreCreateFlags::empty());
    vk::SemaphoreCreateInfo::default().flags(flags)
}
