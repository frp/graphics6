#[macro_use]
extern crate glium;
extern crate time;

use std::f32::consts;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
}

implement_vertex!(Vertex, position);

fn rotate_matrix(v: Vec<f32>) -> [[f32;3];3] {
    let yaw = v[0];
    let pitch = v[1];
    let roll = v[2];
    [
        [yaw.cos()*pitch.cos(), yaw.cos()*pitch.sin()*roll.sin() - yaw.sin()*roll.cos(), yaw.cos()*pitch.sin()*roll.cos() + yaw.sin()*roll.sin()],
        [yaw.sin()*pitch.cos(), yaw.sin()*pitch.sin()*roll.sin() + yaw.cos()*roll.cos(), yaw.sin()*pitch.sin()*roll.cos() - yaw.cos()*roll.sin()],
        [-pitch.sin(), pitch.cos()*roll.sin(), pitch.cos()*roll.cos()]
    ]
}

fn generate_vertex_array() -> Vec<Vertex> {
    let mut res: Vec<Vertex> = Vec::new();

    res.push(Vertex { position: [0.0, 0.0, 0.0_f32] });
    res.push(Vertex { position: [1.0, 0.0, 0.0_f32] });
    res.push(Vertex { position: [1.0, 0.0, 0.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 0.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 0.0_f32] });
    res.push(Vertex { position: [0.0, 1.0, 0.0_f32] });
    res.push(Vertex { position: [0.0, 1.0, 0.0_f32] });
    res.push(Vertex { position: [0.0, 0.0, 0.0_f32] });

    res.push(Vertex { position: [0.0, 0.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 0.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 0.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 2.0_f32] });
    res.push(Vertex { position: [0.0, 1.0, 2.0_f32] });
    res.push(Vertex { position: [0.0, 1.0, 2.0_f32] });
    res.push(Vertex { position: [0.0, 0.0, 2.0_f32] });

    res.push(Vertex { position: [0.0, 0.0, 0.0_f32] });
    res.push(Vertex { position: [0.0, 0.0, 2.0_f32] });
    res.push(Vertex { position: [0.0, 0.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 0.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 0.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 0.0, 0.0_f32] });
    res.push(Vertex { position: [1.0, 0.0, 0.0_f32] });
    res.push(Vertex { position: [0.0, 0.0, 0.0_f32] });

    res.push(Vertex { position: [0.0, 1.0, 0.0_f32] });
    res.push(Vertex { position: [0.0, 1.0, 2.0_f32] });
    res.push(Vertex { position: [0.0, 1.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 2.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 0.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 0.0_f32] });
    res.push(Vertex { position: [0.0, 1.0, 0.0_f32] });

    res.push(Vertex { position: [0.0, 1.0, 0.0_f32] });
    res.push(Vertex { position: [0.5, 1.5, 0.0_f32] });
    res.push(Vertex { position: [0.5, 1.5, 0.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 0.0_f32] });

    res.push(Vertex { position: [0.0, 1.0, 2.0_f32] });
    res.push(Vertex { position: [0.5, 1.5, 2.0_f32] });
    res.push(Vertex { position: [0.5, 1.5, 2.0_f32] });
    res.push(Vertex { position: [1.0, 1.0, 2.0_f32] });

    res.push(Vertex { position: [0.5, 1.5, 0.0_f32] });
    res.push(Vertex { position: [0.5, 1.5, 2.0_f32] });

    res
}

fn generate_axis(axis: [f32;3], radius: f32) -> Vec<Vertex> {
    let unit = 1.0/radius;
    let len = 3.0;
    let mut res: Vec<Vertex> = Vec::new();
    res.push(Vertex { position: [0.0, 0.0, 0.0_f32] });
    res.push(Vertex { position: [axis[0] * len, axis[1] * len, axis[2] * len] });
    let mut pt = 0.0;
    while pt < len {
        pt += unit;
        if axis[0] == 0.0 {
            res.push(Vertex { position: [axis[0] * pt - 0.1, axis[1] * pt, axis[2] * pt] });
            res.push(Vertex { position: [axis[0] * pt + 0.1, axis[1] * pt, axis[2] * pt] });
        }
        else {
            res.push(Vertex { position: [axis[0] * pt, axis[1] * pt - 0.1, axis[2] * pt] });
            res.push(Vertex { position: [axis[0] * pt, axis[1] * pt + 0.1, axis[2] * pt] });
        }
    }
    res
}

fn main() {
    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new().build_glium().unwrap();

    let shape = generate_vertex_array();

    for vertex in shape.clone() {
        println!("{} {} {}", vertex.position[0], vertex.position[1], vertex.position[2]);
    }

    let x_axis = generate_axis([1.0, 0.0, 0_f32], 5.0);
    let y_axis = generate_axis([0.0, 1.0, 0_f32], 5.0);
    let z_axis = generate_axis([0.0, 0.0, 1_f32], 5.0);

    let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
    let vbox = glium::VertexBuffer::new(&display, &x_axis).unwrap();
    let vboy = glium::VertexBuffer::new(&display, &y_axis).unwrap();
    let vboz = glium::VertexBuffer::new(&display, &z_axis).unwrap();

    let indices = glium::index::NoIndices(glium::index::PrimitiveType::LinesList);

    let vertex_shader_src = r#"
        #version 140
        in vec3 position;

        uniform mat4 perspective;
        uniform vec3 translation;
        uniform mat3 rot;

        void main() {
            gl_Position = perspective * vec4(rot * position / 2 + translation, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        uniform vec3 input_color;

        out vec4 color;
        void main() {
            color = vec4(input_color, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let black = [0.0, 0.0, 0.0_f32];
    let blue = [0.0, 0.0, 1.0_f32];
    let green = [0.0, 1.0, 0.0_f32];
    let red = [1.0, 0.0, 0.0_f32];

    let mut rotation_start_time = time::now();
    let mut rotation = [0.0, 0.0, 0.0_f32];

    let mut ypr: [f32;3] = [0.0, 0.0, 0.0_f32];

    loop {
        let mut target = display.draw();

        let diff = (time::now() - rotation_start_time).num_milliseconds() as f32 / 3000.0;

        let rot_vec = (0..3).map(|i| rotation[i] * diff + ypr[i]).collect::<Vec<f32>>();
        let rot = rotate_matrix(rot_vec.clone());

        let perspective = {
            let (width, height) = target.get_dimensions();
            let aspect_ratio = height as f32 / width as f32;

            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;

            let f = 1.0 / (fov / 2.0).tan();

            [
                [f *   aspect_ratio   ,    0.0,              0.0              ,   0.0],
                [         0.0         ,     f ,              0.0              ,   0.0],
                [         0.0         ,    0.0,  (zfar+znear)/(zfar-znear)    ,   1.0],
                [         0.0         ,    0.0, -(2.0*zfar*znear)/(zfar-znear),   0.0],
            ]
        };

        let translation = [0.0, 0.0, 2.0_f32];

        target.clear_color(1.0, 1.0, 1.0, 1.0);
        target.clear_depth(1000.0);

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            line_width: Some(1.0),
            polygon_mode: glium::draw_parameters::PolygonMode::Line,
            //backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
            .. Default::default()
        };

        let params_axis = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            line_width: Some(2.0),
            polygon_mode: glium::draw_parameters::PolygonMode::Line,
            //backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
            .. Default::default()
        };

        target.draw(&vertex_buffer, &indices, &program, &uniform! {perspective: perspective, translation: translation, rot: rot, input_color: black},
                    &params).unwrap();

        target.draw(&vbox, &indices, &program, &uniform! {perspective: perspective, translation: translation, rot: rot, input_color: blue},
                    &params_axis).unwrap();

        target.draw(&vboy, &indices, &program, &uniform! {perspective: perspective, translation: translation, rot: rot, input_color: green},
                    &params_axis).unwrap();

        target.draw(&vboz, &indices, &program, &uniform! {perspective: perspective, translation: translation, rot: rot, input_color: red},
                    &params_axis).unwrap();

        target.finish().unwrap();

        for ev in display.poll_events() {
            match ev {
                glium::glutin::Event::Closed => return,
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Pressed, _, Some(glium::glutin::VirtualKeyCode::Down)) => {
                    ypr = [rot_vec[0], rot_vec[1], rot_vec[2]];
                    rotation_start_time = time::now();
                    rotation = [0.0, 0.0, 1.0_f32];
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Released, _, Some(glium::glutin::VirtualKeyCode::Down)) => {
                    ypr = [rot_vec[0], rot_vec[1], rot_vec[2]];
                    rotation = [0.0, 0.0, 0.0_f32];
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Pressed, _, Some(glium::glutin::VirtualKeyCode::Up)) => {
                    ypr = [rot_vec[0], rot_vec[1], rot_vec[2]];
                    rotation_start_time = time::now();
                    rotation = [0.0, 0.0, -1.0_f32];
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Released, _, Some(glium::glutin::VirtualKeyCode::Up)) => {
                    ypr = [rot_vec[0], rot_vec[1], rot_vec[2]];
                    rotation = [0.0, 0.0, 0.0_f32];
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Pressed, _, Some(glium::glutin::VirtualKeyCode::Left)) => {
                    ypr = [rot_vec[0], rot_vec[1], rot_vec[2]];
                    rotation_start_time = time::now();
                    rotation = [0.0, 1.0, 0.0_f32];
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Released, _, Some(glium::glutin::VirtualKeyCode::Left)) => {
                    ypr = [rot_vec[0], rot_vec[1], rot_vec[2]];
                    rotation = [0.0, 0.0, 0.0_f32];
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Pressed, _, Some(glium::glutin::VirtualKeyCode::Right)) => {
                    ypr = [rot_vec[0], rot_vec[1], rot_vec[2]];
                    rotation_start_time = time::now();
                    rotation = [0.0, -1.0, 0.0_f32];
                },
                glium::glutin::Event::KeyboardInput(glium::glutin::ElementState::Released, _, Some(glium::glutin::VirtualKeyCode::Right)) => {
                    ypr = [rot_vec[0], rot_vec[1], rot_vec[2]];
                    rotation = [0.0, 0.0, 0.0_f32];
                },
                _ => ()
            }
        }
    }
}
