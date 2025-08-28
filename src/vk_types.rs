use std::slice;
use std::sync::Arc;

use ash::vk;
use ash::vk::ExtendsDescriptorSetLayoutCreateInfo;
use {ash_bootstrap as vkb, vk_mem as vma};

pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: vma::Allocation,
    pub image_extent: vk::Extent3D,
    pub image_format: vk::Format,
}

pub struct DescriptorLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    device: Arc<vkb::Device>,
}

impl DescriptorLayoutBuilder {
    pub fn new(device: Arc<vkb::Device>) -> Self {
        Self {
            bindings: Vec::new(),
            device,
        }
    }

    pub fn add_binding(&mut self, binding: u32, ty: vk::DescriptorType) {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_count(1)
                .descriptor_type(ty),
        );
    }

    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn build<'a, T: ExtendsDescriptorSetLayoutCreateInfo + ?Sized + 'a>(
        &self,
        shader_stages: vk::ShaderStageFlags,
        p_next: impl Into<Option<&'a mut T>>,
        flags: impl Into<Option<vk::DescriptorSetLayoutCreateFlags>>,
    ) -> Result<vk::DescriptorSetLayout, anyhow::Error> {
        let flags = flags
            .into()
            .unwrap_or(vk::DescriptorSetLayoutCreateFlags::empty());
        let bindings: Vec<_> = self
            .bindings
            .iter()
            .cloned()
            .map(|b| b.stage_flags(shader_stages))
            .collect();
        let mut info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(flags);
        if let Some(p_next) = p_next.into() {
            info = info.push_next(p_next);
        }

        Ok(unsafe { self.device.create_descriptor_set_layout(&info, None)? })
    }
}

pub struct PoolSizeRatio {
    pub ty: vk::DescriptorType,
    pub ratio: f32,
}

pub struct DescriptorAllocator {
    pool: vk::DescriptorPool,
    device: Arc<vkb::Device>,
}

impl DescriptorAllocator {
    pub fn new(
        device: Arc<vkb::Device>,
        max_sets: u32,
        pool_ratios: &[PoolSizeRatio],
    ) -> Result<Self, anyhow::Error> {
        let mut pool_sizes = Vec::new();
        for ratio in pool_ratios {
            let count = (ratio.ratio * max_sets as f32) as _;
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(ratio.ty)
                    .descriptor_count(count),
            );
        }
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        Ok(Self { device, pool })
    }

    pub fn clear_descriptors(&self) -> Result<(), anyhow::Error> {
        unsafe {
            self.device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::default())?
        }
        Ok(())
    }

    pub fn allocate(
        &self,
        layout: &vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, anyhow::Error> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(slice::from_ref(layout));
        let ds = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)?
                .into_iter()
                .next()
                .unwrap()
        };
        Ok(ds)
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_pool(self.pool, None) };
    }
}
