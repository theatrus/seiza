//! Run the upload reader under an allocation ceiling in a child process.
//! A regression exits only the child, never allocates the claimed pixels.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Capped;
const CEILING: usize = 512 << 20;
static LARGEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        if layout.size() > CEILING {
            std::process::exit(42)
        } else {
            unsafe { System.alloc(layout) }
        }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Capped = Capped;

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut crc = !0_u32;
    for &byte in kind.iter().chain(data) {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 * (crc & 1));
        }
    }
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    png.extend_from_slice(&(!crc).to_be_bytes());
}

fn oversized_png() -> Vec<u8> {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend_from_slice(&200_000_u32.to_be_bytes());
    header.extend_from_slice(&200_000_u32.to_be_bytes());
    header.extend_from_slice(&[16, 6, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &[0x78, 0x9c, 3, 0, 0, 0, 0, 1]);
    chunk(&mut png, b"IEND", &[]);
    png
}

#[test]
fn uploaded_image_budget_is_checked_before_allocating_pixels() {
    const CHILD: &str = "SEIZA_TEST_RASTER_BUDGET_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "uploaded_image_budget_is_checked_before_allocating_pixels",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "bounded upload child failed: {}\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
        eprint!("{}", String::from_utf8_lossy(&result.stderr));
        return;
    }
    let png = oversized_png();
    LARGEST.store(0, Ordering::Relaxed);
    let result = seiza_image_io::raster::decode_oriented(&png);
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(matches!(
        result,
        Err(seiza_image_io::raster::RasterError::Image(
            image::ImageError::Limits(_)
        ))
    ));
    eprintln!("largest allocation request before refusal: {largest} bytes");
}
