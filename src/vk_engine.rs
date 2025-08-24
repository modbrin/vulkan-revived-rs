use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;
use std::{slice, thread};

use ash::vk;
use raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle};
use sdl3::event::{Event as SdlEvent, WindowEvent as SdlWindowEvent};
use sdl3::keyboard::Keycode;
use sdl3::video::Window as SdlWindow;
use sdl3::{Sdl, VideoSubsystem as SdlVideo};
use {ash_bootstrap as vkb, vk_mem as vma};

use crate::{vk_image as vkimg, vk_initializers as vkinit};

const NANOS_IN_SECOND: u64 = 1_000_000_000;

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
    graphics_queue_index: u32,
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
            graphics_queue_index: graphics_queue_index as u32,
            alloc,
        })
    }

    fn destroy(&mut self) {
        self.device.destroy();
        self.instance.destroy();
    }
}

const FRAME_OVERLAP: usize = 2;

#[derive(Debug)]
struct FrameData {
    command_pool: vk::CommandPool,
    main_command_buffer: vk::CommandBuffer,
    image_acquried_semaphore: vk::Semaphore,
    frame_fence: vk::Fence,
}

#[derive(Debug)]
struct FramesState([FrameData; FRAME_OVERLAP]);

impl AsRef<[FrameData]> for FramesState {
    fn as_ref(&self) -> &[FrameData] {
        &self.0
    }
}

impl Deref for FramesState {
    type Target = [FrameData];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FramesState {
    fn new(vulkan: &VulkanState) -> Result<Self, anyhow::Error> {
        let mut frames = Vec::new();
        let cmd_pool_info = vkinit::command_pool_create_info(
            vulkan.graphics_queue_index,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        );
        let fence_info = vkinit::fence_create_info(vk::FenceCreateFlags::SIGNALED);
        let semaphore_info = vkinit::semaphore_create_info(None);
        for _ in 0..FRAME_OVERLAP {
            let command_pool = unsafe { vulkan.device.create_command_pool(&cmd_pool_info, None)? };
            let cmd_buffer_info = vkinit::command_buffer_allocate_info(command_pool, 1);
            let command_buffer =
                unsafe { vulkan.device.allocate_command_buffers(&cmd_buffer_info)? };
            let fence = unsafe { vulkan.device.create_fence(&fence_info, None)? };
            let semaphore = unsafe { vulkan.device.create_semaphore(&semaphore_info, None)? };

            frames.push(FrameData {
                command_pool,
                main_command_buffer: command_buffer
                    .into_iter()
                    .next()
                    .expect("allocated exactly 1"),
                frame_fence: fence,
                image_acquried_semaphore: semaphore,
            });
        }
        Ok(Self(
            frames.try_into().expect("size defined by FRAME_OVERLAP"),
        ))
    }

    fn destroy(&mut self, device: &ash::Device) {
        for frame in &self.0 {
            unsafe {
                device.destroy_command_pool(frame.command_pool, None);
                device.destroy_fence(frame.frame_fence, None);
                device.destroy_semaphore(frame.image_acquried_semaphore, None);
            }
        }
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
    vulkan_state: VulkanState,
    swapchain_state: SwapchainState,
    frames: FramesState,
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
        let frames = FramesState::new(&vulkan)?;
        Ok(Self {
            app,
            sdl,
            vulkan_state: vulkan,
            swapchain_state: swapchain,
            frames,
        })
    }

    pub fn run(&mut self) -> Result<(), anyhow::Error> {
        let mut event_pump = self.sdl.handle.event_pump()?;

        let mut quit = false;
        while !quit {
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

            self.draw()?;
        }

        Ok(())
    }

    fn current_frame(&self) -> &FrameData {
        &self.frames[self.app.frame_number % FRAME_OVERLAP]
    }

    fn draw(&mut self) -> Result<(), anyhow::Error> {
        let frame = self.current_frame();
        let frame_fence = frame.frame_fence;
        let image_acquired_semaphore = frame.image_acquried_semaphore;
        let cmd = frame.main_command_buffer;
        let vk_device = self.vulkan_state.device.deref();
        let sc_device = self.swapchain_state.swapchain.deref();
        let swapchain = self.swapchain_state.swapchain.as_ref();
        let queue = self.vulkan_state.graphics_queue;

        unsafe { vk_device.wait_for_fences(slice::from_ref(&frame_fence), true, NANOS_IN_SECOND)? };
        unsafe { vk_device.reset_fences(slice::from_ref(&frame_fence))? };

        let (swapchain_image_index, needs_resize) = unsafe {
            sc_device.acquire_next_image(
                *swapchain,
                NANOS_IN_SECOND,
                image_acquired_semaphore,
                vk::Fence::null(),
            )?
        };
        unsafe { vk_device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())? };

        let cmd_begin_info =
            vkinit::command_buffer_begin_info(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { vk_device.begin_command_buffer(cmd, &cmd_begin_info)? };

        let swapchain_img = self.swapchain_state.images[swapchain_image_index as usize];
        vkimg::transition_image(
            vk_device,
            cmd,
            swapchain_img,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        let flash = (self.app.frame_number as f32 / 120.0).sin().abs();
        let clear_value = vk::ClearColorValue {
            float32: [0.0, 0.0, flash, 1.0],
        };
        let clear_range = vkinit::image_subresource_range(vk::ImageAspectFlags::COLOR);
        unsafe {
            vk_device.cmd_clear_color_image(
                cmd,
                swapchain_img,
                vk::ImageLayout::GENERAL,
                &clear_value,
                slice::from_ref(&clear_range),
            );
        }
        vkimg::transition_image(
            vk_device,
            cmd,
            swapchain_img,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        unsafe {
            vk_device.end_command_buffer(cmd)?;
        }

        let ready_to_present_semaphore =
            self.swapchain_state.ready_to_present_semaphores[swapchain_image_index as usize];
        let cmd_info = vkinit::command_buffer_submit_info(cmd);
        let signal_info = vkinit::semaphore_submit_info(
            vk::PipelineStageFlags2::ALL_GRAPHICS,
            ready_to_present_semaphore,
        );
        let wait_info = vkinit::semaphore_submit_info(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR,
            image_acquired_semaphore,
        );
        let submit_info = vkinit::submit_info(&cmd_info, &signal_info, &wait_info);

        unsafe { vk_device.queue_submit2(queue, slice::from_ref(&submit_info), frame_fence)? };

        let present_info = vkinit::present_info()
            .swapchains(slice::from_ref(swapchain))
            .wait_semaphores(slice::from_ref(&ready_to_present_semaphore))
            .image_indices(slice::from_ref(&swapchain_image_index));
        let needs_resize = unsafe { sc_device.queue_present(queue, &present_info)? };

        self.app.frame_number += 1;

        Ok(())
    }

    fn destroy(&mut self) {
        unsafe {
            self.vulkan_state.device.device_wait_idle().unwrap();
        }

        self.frames.destroy(&self.vulkan_state.device);
        self.swapchain_state.destroy();
        self.vulkan_state.destroy();
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        self.destroy();
    }
}
