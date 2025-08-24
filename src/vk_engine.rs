use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::Duration;
use std::{slice, thread};

use ash::vk;
use raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle, WindowHandle};
use sdl3::event::{Event as SdlEvent, WindowEvent as SdlWindowEvent};
use sdl3::keyboard::Keycode;
use sdl3::video::Window as SdlWindow;
use sdl3::{Sdl, VideoSubsystem as SdlVideo};
use vk_mem::Alloc;
use {ash_bootstrap as vkb, vk_mem as vma};

use crate::vk_types::AllocatedImage;
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
    alloc: Option<vma::Allocator>,
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
            alloc: Some(alloc),
        })
    }

    fn alloc(&self) -> &vma::Allocator {
        self.alloc.as_ref().unwrap()
    }

    fn destroy(&mut self) {
        self.alloc.take();
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
    del_queue: DeletionQueue,
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

impl DerefMut for FramesState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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
                del_queue: DeletionQueue::new(),
            });
        }
        Ok(Self(
            frames.try_into().expect("size defined by FRAME_OVERLAP"),
        ))
    }

    fn destroy(&mut self, vulkan_state: &VulkanState) {
        for frame in &mut self.0 {
            unsafe {
                vulkan_state
                    .device
                    .destroy_command_pool(frame.command_pool, None);
                vulkan_state.device.destroy_fence(frame.frame_fence, None);
                vulkan_state
                    .device
                    .destroy_semaphore(frame.image_acquried_semaphore, None);
                frame.del_queue.flush();
            }
        }
    }
}

struct SwapchainState {
    surface_format: vk::Format,
    swapchain: vkb::Swapchain,
    // secondary data
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    ready_to_present_semaphores: Vec<vk::Semaphore>,
    // separate draw image
    draw_image: AllocatedImage,
}

impl SwapchainState {
    fn new(vk_state: &VulkanState, size: vk::Extent2D) -> Result<Self, anyhow::Error> {
        let instance = vk_state.instance.clone();
        let device = vk_state.device.clone();

        let draw_image = Self::create_draw_image(vk_state, size)?;

        let surface_format = vk::Format::B8G8R8A8_SRGB;
        let swapchain = Self::create_swapchain(instance, device, surface_format, size)?;
        let mut init = Self {
            surface_format,
            swapchain,
            images: Vec::new(),
            image_views: Vec::new(),
            ready_to_present_semaphores: Vec::new(),
            draw_image,
        };
        init.recreate_swapchain_images()?;
        init.recreate_semaphores(vk_state)?;

        Ok(init)
    }

    fn create_draw_image(
        vulkan_state: &VulkanState,
        size: vk::Extent2D,
    ) -> Result<AllocatedImage, anyhow::Error> {
        let draw_image_format = vk::Format::R16G16B16A16_SFLOAT;
        let draw_image_usages = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let draw_image_extent = vk::Extent3D::default()
            .width(size.width)
            .height(size.height)
            .depth(1);

        let img_create_info =
            vkinit::image_create_info(draw_image_format, draw_image_usages, draw_image_extent);
        let mut img_alloc_info = vma::AllocationCreateInfo::default();
        img_alloc_info.usage = vma::MemoryUsage::AutoPreferDevice;
        img_alloc_info.required_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let (image, allocation) = unsafe {
            vulkan_state
                .alloc()
                .create_image(&img_create_info, &img_alloc_info)?
        };

        let imgview_create_info =
            vkinit::image_view_create_info(draw_image_format, image, vk::ImageAspectFlags::COLOR);
        let image_view = unsafe {
            vulkan_state
                .device
                .create_image_view(&imgview_create_info, None)?
        };

        let draw_image = AllocatedImage {
            image_extent: draw_image_extent,
            image_format: draw_image_format,
            image,
            image_view,
            allocation,
        };
        Ok(draw_image)
    }

    fn destroy_draw_image(&mut self, vulkan_state: &VulkanState) {
        unsafe {
            vulkan_state
                .device
                .destroy_image_view(self.draw_image.image_view, None)
        };
        unsafe {
            vulkan_state
                .alloc()
                .destroy_image(self.draw_image.image, &mut self.draw_image.allocation)
        };
    }

    fn resize_swapchain(
        &mut self,
        vk_state: &VulkanState,
        new_size: vk::Extent2D,
    ) -> Result<(), anyhow::Error> {
        let instance = vk_state.instance.clone();
        let device = vk_state.device.clone();

        self.destroy_semaphores(vk_state);
        self.destroy_swapchain();

        self.swapchain = Self::create_swapchain(instance, device, self.surface_format, new_size)?;
        self.recreate_swapchain_images()?;
        self.recreate_semaphores(vk_state)?;
        Ok(())
    }

    fn recreate_swapchain_images(&mut self) -> Result<(), anyhow::Error> {
        self.images = self.swapchain.get_images()?;
        self.image_views = self.swapchain.get_image_views()?;
        Ok(())
    }

    fn destroy_swapchain(&mut self) {
        self.swapchain.destroy_image_views().unwrap();
        self.swapchain.destroy();

        self.image_views.clear();
        self.images.clear();
    }

    fn recreate_semaphores(&mut self, vk_state: &VulkanState) -> Result<(), anyhow::Error> {
        let mut semaphores = Vec::new();
        for _ in 0..self.images.len() {
            unsafe {
                semaphores.push(
                    vk_state
                        .device
                        .create_semaphore(&vkinit::semaphore_create_info(None), None)?,
                );
            }
        }
        self.ready_to_present_semaphores = semaphores;
        Ok(())
    }

    fn destroy_semaphores(&mut self, vk_state: &VulkanState) {
        for sem in self.ready_to_present_semaphores.drain(..) {
            unsafe {
                vk_state.device.destroy_semaphore(sem, None);
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

    fn destroy(&mut self, vk_state: &VulkanState) {
        self.destroy_draw_image(vk_state);
        self.destroy_swapchain();
        self.destroy_semaphores(vk_state);
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

struct DeletionQueue {
    deletors: VecDeque<Box<dyn FnOnce()>>,
}

impl Debug for DeletionQueue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "deletion queue with {} pending tasks",
            self.deletors.len()
        )
    }
}

impl DeletionQueue {
    fn new() -> Self {
        Self {
            deletors: VecDeque::new(),
        }
    }
    fn push<F: FnOnce() + 'static>(&mut self, f: F) {
        self.deletors.push_back(Box::new(f));
    }
    fn flush(&mut self) {
        while let Some(f) = self.deletors.pop_back() {
            f();
        }
    }
}

pub struct VulkanEngine {
    app: AppState,
    sdl: SdlContext,
    vulkan_state: VulkanState,
    swapchain_state: SwapchainState,
    frames: FramesState,
    del_queue: DeletionQueue,
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
        let swapchain = SwapchainState::new(&vulkan, size)?;
        let frames = FramesState::new(&vulkan)?;
        let del_queue = DeletionQueue::new();
        Ok(Self {
            app,
            sdl,
            vulkan_state: vulkan,
            swapchain_state: swapchain,
            frames,
            del_queue,
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
                continue;
            }

            self.draw()?;
        }

        Ok(())
    }

    fn current_frame_index(&self) -> usize {
        self.app.frame_number % FRAME_OVERLAP
    }

    /// returns VulkanDevice
    fn vk_device(&self) -> &vkb::Device {
        self.vulkan_state.device.deref()
    }

    /// returns SwapchainDevice
    fn sc_device(&self) -> &ash::khr::swapchain::Device {
        self.swapchain_state.swapchain.deref()
    }

    /// returns SwapchainKHR
    fn swapchain(&self) -> &vk::SwapchainKHR {
        self.swapchain_state.swapchain.as_ref()
    }

    fn draw(&mut self) -> Result<(), anyhow::Error> {
        let frame_index = self.current_frame_index();
        let frame = &self.frames[frame_index];
        let frame_fence = frame.frame_fence;
        let image_acquired_semaphore = frame.image_acquried_semaphore;
        let cmd = frame.main_command_buffer;
        let queue = self.vulkan_state.graphics_queue;

        unsafe {
            self.vk_device().wait_for_fences(
                slice::from_ref(&frame_fence),
                true,
                NANOS_IN_SECOND,
            )?
        };

        self.frames[frame_index].del_queue.flush();

        unsafe {
            self.vk_device()
                .reset_fences(slice::from_ref(&frame_fence))?
        };

        let (swapchain_image_index, needs_resize) = unsafe {
            self.sc_device().acquire_next_image(
                *self.swapchain(),
                NANOS_IN_SECOND,
                image_acquired_semaphore,
                vk::Fence::null(),
            )?
        };
        if needs_resize {
            self.app.resize_requested = true;
            return Ok(());
        }
        unsafe {
            self.vk_device()
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?
        };

        let cmd_begin_info =
            vkinit::command_buffer_begin_info(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.vk_device()
                .begin_command_buffer(cmd, &cmd_begin_info)?
        };

        let swapchain_image = self.swapchain_state.images[swapchain_image_index as usize];
        let swapchain_image_extent = self.swapchain_state.swapchain.extent;
        let draw_image = self.swapchain_state.draw_image.image;
        let draw_image_extent = vk::Extent2D::default()
            .width(self.swapchain_state.draw_image.image_extent.width)
            .height(self.swapchain_state.draw_image.image_extent.height);
        vkimg::transition_image(
            self.vk_device(),
            cmd,
            draw_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        self.draw_background(cmd)?;

        vkimg::transition_image(
            self.vk_device(),
            cmd,
            draw_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        vkimg::transition_image(
            self.vk_device(),
            cmd,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        vkimg::copy_image_to_image(
            self.vk_device(),
            cmd,
            draw_image,
            swapchain_image,
            draw_image_extent,
            swapchain_image_extent,
        );
        vkimg::transition_image(
            self.vk_device(),
            cmd,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        unsafe {
            self.vk_device().end_command_buffer(cmd)?;
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

        unsafe {
            self.vk_device()
                .queue_submit2(queue, slice::from_ref(&submit_info), frame_fence)?
        };

        let present_info = vkinit::present_info()
            .swapchains(slice::from_ref(self.swapchain()))
            .wait_semaphores(slice::from_ref(&ready_to_present_semaphore))
            .image_indices(slice::from_ref(&swapchain_image_index));
        let needs_resize = unsafe { self.sc_device().queue_present(queue, &present_info)? };
        if needs_resize {
            self.app.resize_requested = true;
            return Ok(());
        }

        self.app.frame_number += 1;

        Ok(())
    }

    fn draw_background(&mut self, cmd: vk::CommandBuffer) -> Result<(), anyhow::Error> {
        let flash = (self.app.frame_number as f32 / 120.0).sin().abs();
        let clear_value = vk::ClearColorValue {
            float32: [0.0, 0.0, flash, 1.0],
        };
        let clear_range = vkinit::image_subresource_range(vk::ImageAspectFlags::COLOR);
        unsafe {
            self.vk_device().cmd_clear_color_image(
                cmd,
                self.swapchain_state.draw_image.image,
                vk::ImageLayout::GENERAL,
                &clear_value,
                slice::from_ref(&clear_range),
            );
        }
        Ok(())
    }

    fn destroy(&mut self) {
        unsafe {
            self.vulkan_state.device.device_wait_idle().unwrap();
        }

        self.del_queue.flush();
        self.frames.destroy(&self.vulkan_state);
        self.swapchain_state.destroy(&self.vulkan_state);
        self.vulkan_state.destroy();
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        self.destroy();
    }
}
