use wasm_bindgen::prelude::*;
use image::{io::Reader, ImageFormat, DynamicImage};
use image::imageops::resize;

#[wasm_bindgen]
pub fn preprocess(image_data: &[u8]) -> Vec<f32> {
    // Decode image (fallback to JPEG if format detection fails)
    let img = match Reader::with_format(
        std::io::Cursor::new(image_data),
        ImageFormat::from_mime_type("image/jpeg").unwrap_or(ImageFormat::Jpeg)
    ).decode() {
        Ok(img) => img,
        Err(e) => {
            // Log error to console without web-sys
            panic!("Failed to decode image: {}", e);
        }
    };

    // Resize and convert to RGB
    let resized = resize(&img.to_rgb8(), 224, 224, image::imageops::FilterType::Triangle);
    
    // Normalize and flatten to CHW format
    let mut output = Vec::with_capacity(3 * 224 * 224);
    for y in 0..224 {
        for x in 0..224 {
            let pixel = resized.get_pixel(x, y);
            for c in 0..3 {
                output.push((pixel[c] as f32 / 255.0 - [0.485, 0.456, 0.406][c]) / 
                          [0.229, 0.224, 0.225][c]);
            }
        }
    }
    
    output
}
