use std::fs;
use std::path::Path;
use std::process::Command;

const SHADERS_DIR: &str = "./shaders";
const OUT_DIR: &str = "./shaders/out";

fn is_shader_file(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ["comp", "vert", "frag"].into_iter().any(|e| e == ext))
}

fn compile_shader(path: &Path) {
    let name = path.file_name().unwrap().to_str().unwrap();
    let path = path
        .as_os_str()
        .to_str()
        .expect("shader path is valid unicode");
    println!("cargo:rerun-if-changed={}", path);
    let out_path = format!("{}/{}.spv", OUT_DIR, name);
    let res = Command::new("glslangValidator")
        .args(&["-V", path, "-o", out_path.as_str()])
        .output()
        .unwrap();
    if !res.status.success() {
        panic!(
            "shader compilation failed: {}",
            str::from_utf8(&res.stdout).unwrap()
        )
    }
}

pub fn main() {
    let _ = fs::create_dir_all(OUT_DIR);
    for entry in fs::read_dir(SHADERS_DIR).unwrap() {
        let entry = entry.unwrap();
        if is_shader_file(&entry.path()) {
            compile_shader(&entry.path());
        }
    }
}
