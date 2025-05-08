use wasm_bindgen::prelude::*;
use image::io::Reader as ImageReader;
use image::imageops::resize;
use image::{DynamicImage, GenericImageView, Rgba};
use image::ImageBuffer;

#[wasm_bindgen]
pub fn preprocess_image(image_data: &[u8]) -> Vec<f32> {
    let img = ImageReader::new(std::io::Cursor::new(image_data))
        .with_guessed_format()
        .expect("Failed to guess image format")
        .decode()
        .expect("Failed to decode image");

    // Convert to RGB if needed and resize
    let img_rgb = img.into_rgb8();
    let resized = resize(&img_rgb, 224, 224, image::imageops::FilterType::Triangle);
    
    // Convert to DynamicImage for normalization
    let resized_dynamic = DynamicImage::ImageRgb8(resized);
    let normalized = normalize_image(resized_dynamic);

    // Convert to CHW format (3x224x224) and flatten
    let mut output = Vec::with_capacity(3 * 224 * 224);
    for c in 0..3 {
        for y in 0..224 {
            for x in 0..224 {
                let pixel = normalized.get_pixel(x, y);
                output.push((pixel[c] as f32 / 255.0 - MEAN[c]) / STD[c]);
            }
        }
    }
    output
}

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn normalize_image(img: DynamicImage) -> DynamicImage {
    img
}
