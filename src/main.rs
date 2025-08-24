use crate::vk_engine::VulkanEngine;

mod vk_engine;
mod vk_image;
mod vk_initializers;

fn main() {
    tracing_subscriber::fmt::init();

    VulkanEngine::init().unwrap().run().unwrap();
}
