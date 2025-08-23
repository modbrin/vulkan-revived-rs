use std::thread;
use std::time::Duration;

use sdl3::event::{Event as SdlEvent, WindowEvent as SdlWindowEvent};
use sdl3::keyboard::Keycode;
use sdl3::video::Window as SdlWindow;
use sdl3::{Sdl, VideoSubsystem as SdlVideo};

pub struct AppState {
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
    app_state: AppState,
    sdl_context: SdlContext,
}

impl VulkanEngine {
    pub fn init() -> Result<Self, anyhow::Error> {
        Ok(Self {
            app_state: AppState::new()?,
            sdl_context: SdlContext::new()?,
        })
    }

    pub fn run(&mut self) -> Result<(), anyhow::Error> {
        let mut event_pump = self.sdl_context.handle.event_pump()?;

        let mut quit = false;
        loop {
            for event in event_pump.poll_iter() {
                match event {
                    SdlEvent::Quit { .. } => {
                        quit = true;
                    }
                    SdlEvent::Window { win_event, .. } => match win_event {
                        SdlWindowEvent::Minimized => self.app_state.stop_rendering = true,
                        SdlWindowEvent::Maximized => self.app_state.stop_rendering = false,
                        SdlWindowEvent::Resized(w, h) => self.app_state.resize_requested = true,
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

            if self.app_state.stop_rendering {
                thread::sleep(Duration::from_millis(100));
            }

            if quit {
                break;
            }
        }

        Ok(())
    }
}
