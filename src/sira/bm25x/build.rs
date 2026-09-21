// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

fn main() {
    #[cfg(feature = "cuda")]
    {
        // cudarc uses dynamic-loading (dlopen/LoadLibrary at runtime), so these
        // link search paths are only needed if cudarc ever links statically.
        // We add them opportunistically — missing paths are silently skipped.

        // Custom CUDA_PATH (works on all platforms)
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            for suffix in &["lib64/stubs", "lib64", "lib/x64", "lib"] {
                let p = format!("{}/{}", cuda_path, suffix);
                if std::path::Path::new(&p).exists() {
                    println!("cargo:rustc-link-search=native={}", p);
                }
            }
        }

        // Linux standard paths
        #[cfg(target_os = "linux")]
        {
            let candidates = [
                "/usr/local/cuda/lib64/stubs",
                "/usr/local/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu",
            ];
            for path in &candidates {
                if std::path::Path::new(path).exists() {
                    println!("cargo:rustc-link-search=native={}", path);
                }
            }
            // Versioned cuda installs: /usr/local/cuda-*/lib64/stubs
            if let Ok(entries) = std::fs::read_dir("/usr/local") {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    if name.starts_with("cuda-") {
                        let stubs = format!("/usr/local/{}/lib64/stubs", name);
                        if std::path::Path::new(&stubs).exists() {
                            println!("cargo:rustc-link-search=native={}", stubs);
                        }
                    }
                }
            }
        }

        // Windows standard paths
        #[cfg(target_os = "windows")]
        {
            let program_files =
                std::env::var("ProgramFiles").unwrap_or_else(|_| "C:\\Program Files".to_string());
            let base = format!("{}\\NVIDIA GPU Computing Toolkit\\CUDA", program_files);
            if let Ok(entries) = std::fs::read_dir(&base) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    let lib = p.join("lib").join("x64");
                    if lib.exists() {
                        println!("cargo:rustc-link-search=native={}", lib.display());
                    }
                }
            }
        }
    }
}
