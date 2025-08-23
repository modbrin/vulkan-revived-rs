use crate::engine::VulkanEngine;

mod engine;

fn main() {
    VulkanEngine::init().unwrap().run().unwrap();
}
