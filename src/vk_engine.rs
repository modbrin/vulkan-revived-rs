use std::sync::Arc;
use std::thread;
use std::time::Duration;

use ash::vk;
use raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle};
use sdl3::event::{Event as SdlEvent, WindowEvent as SdlWindowEvent};
use sdl3::keyboard::Keycode;
use sdl3::video::Window as SdlWindow;
use sdl3::{Sdl, VideoSubsystem as SdlVideo};
use {ash_bootstrap as vkb, vk_mem as vma};

use crate::vk_initializers as vkinit;

trait OrderedDestroy {
    fn destroy(&mut self);
}

struct AppState {
    frame_number: usize,
    stop_rendering: bool,
    resize_requested: bool,
    render_scale: f32,
}

impl AppState {
    pub fn new() -> Result<Self, anyhow::Error> {
        Ok(Self {
            frame_number: 0,
            stop_rendering: false,
            resize_requested: false,
            render_scale: 1.0,
        })
    }
}

pub struct VulkanState {
    instance: Arc<vkb::Instance>,
    device: Arc<vkb::Device>,
    graphics_queue: vk::Queue,
    graphics_queue_index: usize,
    alloc: vma::Allocator,
}

impl VulkanState {
    pub fn new(
        window_hnd: WindowHandle,
        display_hnd: DisplayHandle,
    ) -> Result<Self, anyhow::Error> {
        let instance = vkb::InstanceBuilder::new(Some((window_hnd, display_hnd)))
            .app_name("vulkan-revived")
            .engine_name("vulkan-revived")
            .request_validation_layers(true)
            .require_api_version(vk::make_api_version(0, 1, 3, 0))
            .use_default_tracing_messenger()
            .build()?;
        let vk12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true);
        let vk13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let physical_device = vkb::PhysicalDeviceSelector::new(instance.clone())
            .preferred_device_type(vkb::PreferredDeviceType::Discrete)
            .add_required_extension_feature(vk12_features)
            .add_required_extension_feature(vk13_features)
            .select()?;
        let device = Arc::new(vkb::DeviceBuilder::new(physical_device, instance.clone()).build()?);
        let (graphics_queue_index, graphics_queue) = device.get_queue(vkb::QueueType::Graphics)?;

        let alloc = unsafe {
            vma::Allocator::new(vma::AllocatorCreateInfo::new(
                vkb::Instance::as_ref(&instance),
                vkb::Device::as_ref(&device),
                device.physical_device().as_ref().clone(),
            ))?
        };

        Ok(Self {
            instance,
            device,
            graphics_queue,
            graphics_queue_index,
            alloc,
        })
    }
}

impl OrderedDestroy for VulkanState {
    fn destroy(&mut self) {
        self.device.destroy();
        self.instance.destroy();
    }
}

struct SwapchainState {
    instance: Arc<vkb::Instance>,
    device: Arc<vkb::Device>,
    surface_format: vk::Format,
    swapchain: vkb::Swapchain,
    // secondary data
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    ready_to_present_semaphores: Vec<vk::Semaphore>,
}

impl SwapchainState {
    fn new(
        instance: Arc<vkb::Instance>,
        device: Arc<vkb::Device>,
        size: vk::Extent2D,
    ) -> Result<Self, anyhow::Error> {
        let surface_format = vk::Format::B8G8R8A8_SRGB;
        let swapchain =
            Self::create_swapchain(instance.clone(), device.clone(), surface_format, size)?;
        let mut init = Self {
            instance,
            device,
            surface_format,
            swapchain,
            images: Vec::new(),
            image_views: Vec::new(),
            ready_to_present_semaphores: Vec::new(),
        };
        init.recreate_images()?;
        init.recreate_semaphores()?;
        Ok(init)
    }

    fn resize_swapchain(&mut self, size: vk::Extent2D) -> Result<(), anyhow::Error> {
        self.destroy();

        self.swapchain = Self::create_swapchain(
            self.instance.clone(),
            self.device.clone(),
            self.surface_format,
            size,
        )?;
        self.recreate_images()?;
        self.recreate_semaphores()?;
        Ok(())
    }

    fn recreate_images(&mut self) -> Result<(), anyhow::Error> {
        self.images = self.swapchain.get_images()?;
        self.image_views = self.swapchain.get_image_views()?;
        Ok(())
    }

    fn recreate_semaphores(&mut self) -> Result<(), anyhow::Error> {
        let mut semaphores = Vec::new();
        for _ in 0..self.images.len() {
            unsafe {
                semaphores.push(
                    self.device
                        .create_semaphore(&vkinit::semaphore_create_info(None), None)?,
                );
            }
        }
        self.ready_to_present_semaphores = semaphores;
        Ok(())
    }

    fn destroy_semaphores(&mut self) {
        for sem in self.ready_to_present_semaphores.drain(..) {
            unsafe {
                self.device.destroy_semaphore(sem, None);
            }
        }
    }

    fn create_swapchain(
        instance: Arc<vkb::Instance>,
        device: Arc<vkb::Device>,
        surface_format: vk::Format,
        size: vk::Extent2D,
    ) -> Result<vkb::Swapchain, anyhow::Error> {
        let mut format = vk::SurfaceFormat2KHR::default();
        format.surface_format.color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        format.surface_format.format = surface_format;

        let builder = vkb::SwapchainBuilder::new(instance, device)
            .desired_format(format)
            .desired_present_mode(vk::PresentModeKHR::FIFO)
            .desired_size(size)
            .add_image_usage_flags(vk::ImageUsageFlags::TRANSFER_DST);

        Ok(builder.build()?)
    }
}

impl OrderedDestroy for SwapchainState {
    fn destroy(&mut self) {
        self.destroy_semaphores();
        self.swapchain.destroy_image_views().unwrap();
        self.swapchain.destroy();

        self.image_views.clear();
        self.images.clear();
    }
}

pub struct SdlContext {
    handle: Sdl,
    video: SdlVideo,
    window: SdlWindow,
}

impl SdlContext {
    pub fn new() -> Result<Self, anyhow::Error> {
        let sdl = sdl3::init()?;
        let video_subsystem = sdl.video()?;
        let window = video_subsystem
            .window("Vulkan Revived", 1920, 1080)
            .vulkan()
            .resizable()
            .build()?;
        Ok(Self {
            handle: sdl,
            video: video_subsystem,
            window,
        })
    }
}

pub struct VulkanEngine {
    app: AppState,
    sdl: SdlContext,
    vulkan: VulkanState,
    swapchain: SwapchainState,
}

impl VulkanEngine {
    pub fn init() -> Result<Self, anyhow::Error> {
        let size = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let app = AppState::new()?;
        let sdl = SdlContext::new()?;
        let window_hnd = sdl.window.window_handle().map_err(anyhow::Error::msg)?;
        let display_hnd = sdl.window.display_handle().map_err(anyhow::Error::msg)?;
        let vulkan = VulkanState::new(window_hnd, display_hnd)?;
        let swapchain = SwapchainState::new(vulkan.instance.clone(), vulkan.device.clone(), size)?;
        Ok(Self {
            app,
            sdl,
            vulkan,
            swapchain,
        })
    }

    pub fn run(&mut self) -> Result<(), anyhow::Error> {
        let mut event_pump = self.sdl.handle.event_pump()?;

        let mut quit = false;
        loop {
            for event in event_pump.poll_iter() {
                match event {
                    SdlEvent::Quit { .. } => {
                        quit = true;
                    }
                    SdlEvent::Window { win_event, .. } => match win_event {
                        SdlWindowEvent::Minimized => self.app.stop_rendering = true,
                        SdlWindowEvent::Maximized => self.app.stop_rendering = false,
                        SdlWindowEvent::Resized(w, h) => self.app.resize_requested = true,
                        _ => (),
                    },
                    SdlEvent::KeyDown { keycode, .. } => match keycode {
                        Some(Keycode::Left) => println!("left button pushed"),
                        Some(Keycode::Right) => println!("right button pushed"),
                        _ => (),
                    },
                    _ => (),
                }
            }

            if self.app.stop_rendering {
                thread::sleep(Duration::from_millis(100));
            }

            if quit {
                break;
            }
        }

        Ok(())
    }
}

impl OrderedDestroy for VulkanEngine {
    fn destroy(&mut self) {
        unsafe {
            self.vulkan.device.device_wait_idle().unwrap();
        }

        self.swapchain.destroy();
        self.vulkan.destroy();
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        self.destroy();
    }
}
