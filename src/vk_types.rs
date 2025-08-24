use ash::vk;
use vk_mem as vma;

pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: vma::Allocation,
    pub image_extent: vk::Extent3D,
    pub image_format: vk::Format,
}
