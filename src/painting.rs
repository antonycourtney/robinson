
extern crate freetype as ft;

use layout::{AnonymousBlock, BlockNode, InlineNode, LayoutBox, Rect};
use css::{Value, Color};
use std::cmp;

pub struct Canvas {
    pub pixels: Vec<Color>,
    pub width: usize,
    pub height: usize,
}

const TEST_FONT: &'static str = "../freetype-rs/examples/assets/FiraSans-Regular.ttf";

/// Paint a tree of LayoutBoxes to an array of pixels.
pub fn paint(layout_root: &LayoutBox, bounds: Rect) -> Canvas {
    // initialize freetype:
    let library = ft::Library::init().unwrap();
    let face = library.new_face(TEST_FONT, 0).unwrap();


    let display_list = build_display_list(layout_root);
    let mut canvas = Canvas::new(bounds.width as usize, bounds.height as usize);
    for item in display_list {
        canvas.paint_item(&item);
    }
    canvas.paint_str(&face, 120, 200, "Lorem Ipsum Lorem Ipsum.");
    return canvas;
}

#[derive(Debug)]
pub enum DisplayCommand {
    SolidColor(Color, Rect),
}

pub type DisplayList = Vec<DisplayCommand>;

pub fn build_display_list(layout_root: &LayoutBox) -> DisplayList {
    let mut list = Vec::new();
    render_layout_box(&mut list, layout_root);
    return list;
}

fn render_layout_box(list: &mut DisplayList, layout_box: &LayoutBox) {
    render_background(list, layout_box);
    render_borders(list, layout_box);
    for child in &layout_box.children {
        render_layout_box(list, child);
    }
}

fn render_background(list: &mut DisplayList, layout_box: &LayoutBox) {
    get_color(layout_box, "background").map(|color|
        list.push(DisplayCommand::SolidColor(color, layout_box.dimensions.border_box())));
}

fn render_borders(list: &mut DisplayList, layout_box: &LayoutBox) {
    let color = match get_color(layout_box, "border-color") {
        Some(color) => color,
        _ => return
    };

    let d = &layout_box.dimensions;
    let border_box = d.border_box();

    // Left border
    list.push(DisplayCommand::SolidColor(color, Rect {
        x: border_box.x,
        y: border_box.y,
        width: d.border.left,
        height: border_box.height,
    }));

    // Right border
    list.push(DisplayCommand::SolidColor(color, Rect {
        x: border_box.x + border_box.width - d.border.right,
        y: border_box.y,
        width: d.border.right,
        height: border_box.height,
    }));

    // Top border
    list.push(DisplayCommand::SolidColor(color, Rect {
        x: border_box.x,
        y: border_box.y,
        width: border_box.width,
        height: d.border.top,
    }));

    // Bottom border
    list.push(DisplayCommand::SolidColor(color, Rect {
        x: border_box.x,
        y: border_box.y + border_box.height - d.border.bottom,
        width: border_box.width,
        height: d.border.bottom,
    }));
}

/// Return the specified color for CSS property `name`, or None if no color was specified.
fn get_color(layout_box: &LayoutBox, name: &str) -> Option<Color> {
    match layout_box.box_type {
        BlockNode(style) | InlineNode(style) => match style.value(name) {
            Some(Value::ColorValue(color)) => Some(color),
            _ => None
        },
        AnonymousBlock => None
    }
}

/**
 * compose two pixels, with alpha blending
 *
 * Based on "Image Compositing Fundamentals" by Alvy Ray Smith,
 * http://alvyray.com/Memos/CG/Microsoft/4_comp.pdf
 *
 */

// approximation of multiplying a and b if we interpret each quantity as a representation
// of a real number on interval [0.0, 1.0]
fn int_mult(a: u8, b: u8) -> u8 {
    let t : u32 = a as u32 * b as u32;
    let ret = ((t >> 8) + t) >> 8;
    ret as u8
}

#[allow(dead_code)]
fn int_lerp(p: u8, q: u8, a: u8) -> u8 {
    let ret = p + int_mult(a,q - p);
    ret
}

#[allow(dead_code)]
fn int_pre_lerp(p: u8, q:u8, a: u8) -> u8 {
    let ret = p + q + int_mult(a,p);
    ret
}

// over for single channel:
#[allow(dead_code)]
fn over_channel(a: u8,alpha: u8,b: u8,beta: u8) -> u8 {
    let c_prime = int_lerp(int_mult(a,alpha),b,beta);
    c_prime
}

// simple but somewhat inefficient over for a single channel:
fn over_channel_simple(a: u8,alpha: u8,b: u8,beta: u8) -> u8 {
    // If we interpret all args as 8 bit representation of floating point values between 0 and 1:
    // res = beta * b + (1 - beta) * alpha * a
    //     = beta * b + alpha * a - alpha * beta * a
    // (Note: The above equations are based on the floating point interpretation)
    let c_prime = int_mult(beta,b) + int_mult(alpha,a) - int_mult(int_mult(alpha,beta),a);
    c_prime
}

fn over(a: Color, b: Color) -> Color {
    let red = over_channel_simple(a.r,a.a,b.r,b.a);
    let green = over_channel_simple(a.g,a.a,b.g,b.a);
    let blue = over_channel_simple(a.b,a.a,b.b,b.a);
    // from simple formulation in paper:
    // beta + alpha - alpha * beta
    // let c_alpha = b.a as u32 + a.a as u32 - int_mult(b.a,a.a) as u32;
    let c_alpha = 255;
    let ret = Color { r: red, g: green, b: blue, a: c_alpha as u8 };
    ret
}


impl Canvas {
    /// Create a blank canvas
    fn new(width: usize, height: usize) -> Canvas {
        let white = Color { r: 255, g: 255, b: 255, a: 255 };
        return Canvas {
            pixels: vec![white; width * height],
            width: width,
            height: height,
        }
    }

    fn paint_item(&mut self, item: &DisplayCommand) {
        match item {
            &DisplayCommand::SolidColor(color, rect) => {
                // Clip the rectangle to the canvas boundaries.
                let x0 = rect.x.clamp(0.0, self.width as f32) as usize;
                let y0 = rect.y.clamp(0.0, self.height as f32) as usize;
                let x1 = (rect.x + rect.width).clamp(0.0, self.width as f32) as usize;
                let y1 = (rect.y + rect.height).clamp(0.0, self.height as f32) as usize;

                for y in y0 .. y1 {
                    for x in x0 .. x1 {
                        // TODO: alpha compositing with existing pixel
                        self.pixels[y * self.width + x] = color;
                    }
                }
            }
        }
    }


    // render a ft::Bitmap on to Canvas
    // adapted from draw_bitmap in single_glyph example in freetype-rs
    fn paint_ft_bitmap(&mut self, color: Color, bitmap: ft::Bitmap, x: usize, y: usize) {
        let mut p = 0;
        let mut q = 0;
        let w = bitmap.width() as usize;
        let x_max = x + w;
        let y_max = y + bitmap.rows() as usize;

        println!("w: {}, x_max: {}, rows: {}, y_max: {}", w, x_max, bitmap.rows(), y_max );
        println!("iterating: x: {}, x_max: {}, y: {}, y_max: {}", x, x_max, y, y_max );
        for i in x .. oclamp(0,self.width-1,x_max) {
            for j in y .. oclamp(0,self.height-1,y_max) {
                let under = self.pixels[j * self.width + i];    // current pixel
                let level = bitmap.buffer()[q * w + p];
                self.pixels[j * self.width + i] = over(under,Color { a: level, .. color });
                q += 1;
            }
            q = 0;
            p += 1;
        }
    }

    fn paint_str(&mut self, face: &ft::Face, x: i64, y: i64, s: &str) {
        let mut transform = ft::Matrix { xx: 0x10000, xy: 0, yx: 0, yy: 0x10000 };  // identity
        let mut pos = ft::Vector { x: x * 64, y: y * 64 };

        face.set_char_size(24 * 64, 0, 120, 0).unwrap();            // 40pt at 110 dpi

        for ch in s.chars() {
            face.set_transform( &mut transform, &mut pos );
            face.load_char(ch as usize, ft::face::RENDER).unwrap();
            let glyph_slot = face.glyph();
            let bitmap = glyph_slot.bitmap();
            // dump_ft_bitmap(bitmap);
            println!("glyph: bitmap_left: {}, bitmap_top: {}", glyph_slot.bitmap_left(),glyph_slot.bitmap_top());
            let gx = glyph_slot.bitmap_left() as usize;
            let gy = (y - (glyph_slot.bitmap_top() as i64 - y)) as usize;
            let pen = Color { r: 0, g: 0, b: 0, a: 255 };
            self.paint_ft_bitmap(pen, bitmap, gx, gy);

            let adv = glyph_slot.advance();
            let gm = glyph_slot.metrics();
            println!("glyph metrics: height: {}, horiBearingY: {}", gm.height, gm.horiBearingY );
            println!("advance: {}, {}", adv.x, adv.y);
            println!("before advance: {}, {}", pos.x, pos.y);
            pos.x += adv.x;
            pos.y += adv.y;
            println!("after advance: {}, {}", pos.x, pos.y);
        }
    }


    // paint a string, advancing the position after each character:
/*
    fn paint_str(&mut self, face: &ft::Face, s: &str) {

        face.set_char_size(64 * 64, 0, 110, 0).unwrap();            // 40pt at 110 dpi

        face.load_char(ch as usize, ft::face::RENDER).unwrap();
        let glyph = face.glyph();
        let bitmap = glyph.bitmap();
        // dump_ft_bitmap(bitmap);
        println!("glyph: bitmap_left: {}, bitmap_top: {}", glyph.bitmap_left(),glyph.bitmap_top());
        let x = glyph.bitmap_left() as usize;
        let y = (glyph.bitmap_top() - bitmap.rows()) as usize;
        let pen = Color { r: 0, g: 0, b: 0, a: 255 };
        self.paint_ft_bitmap(pen, bitmap, x, y);
    }
*/


}

trait Clamp {
    fn clamp(self, lower: Self, upper: Self) -> Self;
}
impl Clamp for f32 {
    fn clamp(self, lower: f32, upper: f32) -> f32 {
        self.max(lower).min(upper)
    }
}

/*
 * A generic three-argument version of clamp for any value in Ord.
 */
fn oclamp<T: Ord>(lower: T, upper: T, val: T) -> T {
    cmp::min(cmp::max(val,lower),upper)
}

/*
 An attempt to define a generic OClamp trait for all <T>: Ord so we can use method syntax:
 Unfortunately, can't work out how to implement; the obvious way (giving a default impl in the trait)
 yields a bunch of inscrutable type errors
trait OClamp: Ord {
    fn oclamp(self, lower: Self, upper: Self) -> Self;
}
*/
