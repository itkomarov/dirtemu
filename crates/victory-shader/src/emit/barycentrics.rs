use super::bitcode::{Attr, Term, V};
use super::container::SigElement;
use super::shader::{Geometry, Shader};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

pub(super) struct Varying {
    pub output: usize,
    pub source: Option<(usize, u8)>,
}

pub(super) fn append(s: &mut Shader, wanted: &[SigElement]) -> Result<Vec<Varying>, ShaderError> {
    let mut varyings = Vec::new();
    for element in wanted {
        let source = if element.name == "VICTORY_BARYCENTRICS" {
            None
        } else if let Some(rest) = element.name.strip_prefix("VICTORY_VERTEX_") {
            let (vertex, semantic) = rest
                .split_once('_')
                .ok_or_else(|| unsupported("a software vertex semantic"))?;
            let vertex: u8 = vertex
                .parse()
                .map_err(|_| unsupported("a software vertex index"))?;
            if vertex > 2 {
                return Err(unsupported("a software vertex index"));
            }
            let input = s
                .inputs
                .iter()
                .position(|input| input.name == semantic && input.index == element.index)
                .ok_or_else(|| {
                    unsupported("a software vertex attribute missing from the vertex stage")
                })?;
            if s.inputs[input].cols != element.cols || s.inputs[input].comp != element.comp {
                return Err(unsupported("software vertex attribute types that differ"));
            }
            Some((input, vertex))
        } else {
            continue;
        };
        let row = element
            .row
            .ok_or_else(|| unsupported("a software interpolation without a register"))?;
        if row >= 32 {
            return Err(unsupported(
                "software barycentrics exceeding 32 signature rows",
            ));
        }
        for output in 0..s.outputs.len() {
            let old = &s.outputs[output];
            if old.row == Some(row)
                && old.col < element.col + element.cols
                && element.col < old.col + old.cols
            {
                if wanted
                    .iter()
                    .any(|input| input.name == old.name && input.index == old.index)
                {
                    return Err(unsupported(
                        "software interpolation overlapping a pixel input",
                    ));
                }
                let spare = s
                    .outputs
                    .iter()
                    .chain(wanted)
                    .filter_map(|e| e.row)
                    .max()
                    .map_or(0, |row| row + 1);
                if spare >= 32 {
                    return Err(unsupported(
                        "software barycentrics exceeding 32 signature rows",
                    ));
                }
                s.outputs[output].row = Some(spare);
            }
        }
        let output = s.outputs.len();
        s.outputs.push(SigElement {
            row: Some(row),
            usage: (1 << element.cols) - 1,
            ..element.clone()
        });
        varyings.push(Varying { output, source });
    }
    Ok(varyings)
}

pub(super) fn store(s: &mut Shader, output: usize, column: u8, value: V) {
    let name = if s.outputs[output].comp == 9 {
        "storeOutput.f32"
    } else {
        "storeOutput.i32"
    };
    let args = [s.m.i32(output as u32), s.m.i32(0), s.m.i8(column), value];
    s.op(s.entry, name, Attr::NoUnwind, s.t.void, 5, &args);
}

pub(super) fn weight(s: &mut Shader, vertex: usize, column: usize) -> V {
    s.m.f32(if vertex == column { 1f32.to_bits() } else { 0 })
}

fn unsupported(what: &'static str) -> ShaderError {
    ShaderError::Unsupported { offset: 0, what }
}

pub fn barycentric_gs(vs: &[u8], ps: &[u8]) -> Result<Option<Vec<u8>>, ShaderError> {
    let pixel = Container::parse(ps)?;
    let wanted = match pixel.part(FourCC::ISG1) {
        Some(part) => super::signature(part.data)?,
        None => return Ok(None),
    };
    if !wanted
        .iter()
        .any(|e| e.name == "VICTORY_BARYCENTRICS" || e.name.starts_with("VICTORY_VERTEX_"))
    {
        return Ok(None);
    }
    let vertex = Container::parse(vs)?;
    let outputs = vertex
        .part(FourCC::OSG1)
        .ok_or_else(|| unsupported("a vertex stage without output signature"))?;
    let outputs = super::signature(outputs.data)?;
    let profile = super::rect::profile_of(vs)
        .ok_or_else(|| unsupported("a vertex shader with an unknown profile"))?;
    build(outputs, &wanted, profile).map(Some)
}

fn load(s: &mut Shader, input: usize, column: u8, vertex: u32) -> V {
    let (name, ty) = if s.inputs[input].comp == 9 {
        ("loadInput.f32", s.t.f32)
    } else {
        ("loadInput.i32", s.t.i32)
    };
    let args = [
        s.m.i32(input as u32),
        s.m.i32(0),
        s.m.i8(column),
        s.m.i32(vertex),
    ];
    s.op(s.entry, name, Attr::ReadNone, ty, 4, &args)
}

fn build(
    mut outputs: Vec<SigElement>,
    wanted: &[SigElement],
    profile: super::ShaderProfile,
) -> Result<Vec<u8>, ShaderError> {
    for output in &mut outputs {
        output.interp = match (output.kind, output.comp) {
            (3, _) => 4,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
        output.usage = (1 << output.cols) - 1;
    }
    let mut s = Shader::new(false, profile);
    s.geometry = Some(Geometry { max_vertices: 3 });
    s.inputs = outputs.clone();
    s.outputs = outputs;
    let varying = append(&mut s, wanted)?;
    let primitive = wanted
        .iter()
        .find(|e| e.kind == 10)
        .filter(|_| !s.outputs.iter().any(|e| e.kind == 10))
        .map(|element| {
            let row = element.row.unwrap_or_else(|| {
                s.outputs
                    .iter()
                    .filter_map(|e| e.row)
                    .max()
                    .map_or(0, |row| row + 1)
            });
            let id = s.outputs.len();
            s.outputs.push(SigElement {
                row: Some(row),
                interp: 1,
                usage: 1,
                ..element.clone()
            });
            id
        });
    if s.outputs.iter().filter_map(|e| e.row).any(|row| row >= 32) {
        return Err(unsupported(
            "software barycentrics exceeding 32 signature rows",
        ));
    }
    let mut inputs = Vec::new();
    for id in 0..s.inputs.len() {
        let mut columns = Vec::new();
        for column in 0..s.inputs[id].cols {
            columns.push([0, 1, 2].map(|vertex| load(&mut s, id, column, vertex)));
        }
        inputs.push(columns);
    }
    let number = primitive.map(|id| {
        (
            id,
            s.op(
                s.entry,
                "primitiveID.i32",
                Attr::ReadNone,
                s.t.i32,
                108,
                &[],
            ),
        )
    });
    let stream = s.m.i8(0);
    for vertex in 0..3 {
        for (output, columns) in inputs.iter().enumerate() {
            for (column, values) in columns.iter().enumerate() {
                store(&mut s, output, column as u8, values[vertex]);
            }
        }
        for varying in &varying {
            for column in 0..s.outputs[varying.output].cols as usize {
                let value = match varying.source {
                    Some((input, from)) => inputs[input][column][from as usize],
                    None => weight(&mut s, vertex, column),
                };
                store(&mut s, varying.output, column as u8, value);
            }
        }
        if let Some((id, number)) = number {
            store(&mut s, id, 0, number);
        }
        s.op(
            s.entry,
            "emitStream",
            Attr::NoUnwind,
            s.t.void,
            97,
            &[stream],
        );
    }
    s.m.terminate(s.entry, Term::Ret);
    Ok(s.finish())
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::bitcode::Bin;
    use super::super::host::{Draw, Host};
    use super::super::{HostShader, ShaderProfile};
    use super::*;

    fn element(name: &str, row: u32, cols: u8, kind: u8, mode: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            row: Some(row),
            cols,
            kind,
            interp: mode,
            col: 0,
            comp: 9,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    fn wrap(stage: crate::native::HwStage, shader: Shader) -> HostShader {
        HostShader {
            stage,
            profile: shader.profile,
            dxil: shader.finish(),
            descriptor_aliases: vec![],
            direct_heap: false,
            mesh_fallback: None,
            user_data: vec![],
            samplers: vec![],
            static_samplers: vec![],
            root_cbvs: vec![],
            root_srvs: vec![],
            root_uavs: vec![],
            root_indexed_views: vec![],
        }
    }

    fn shaders(mode: u8, profile: ShaderProfile) -> (HostShader, HostShader) {
        let mut vs = Shader::new(false, profile);
        vs.inputs = vec![
            element("POSITION", 0, 4, 0, 0),
            element("TEXCOORD", 1, 4, 0, 0),
        ];
        vs.outputs = vec![
            element("SV_Position", 0, 4, 3, 4),
            element("TEXCOORD", 1, 4, 0, 2),
            element("UNUSED", 2, 4, 0, 2),
        ];
        for (input, output) in [(0, 0), (1, 1), (1, 2)] {
            for column in 0..4 {
                let args = [
                    vs.m.i32(input),
                    vs.m.i32(0),
                    vs.m.i8(column),
                    vs.m.undef(vs.t.i32),
                ];
                let value = vs.op(
                    vs.entry,
                    "loadInput.f32",
                    Attr::ReadNone,
                    vs.t.f32,
                    4,
                    &args,
                );
                store(&mut vs, output as usize, column, value);
            }
        }
        vs.m.terminate(vs.entry, Term::Ret);
        let mut ps = Shader::new(true, profile);
        ps.inputs = vec![
            element("SV_Position", 0, 4, 3, 4),
            element("TEXCOORD", 1, 4, 0, 1),
            element("VICTORY_BARYCENTRICS", 2, 3, 0, mode),
            element("VICTORY_VERTEX_0_TEXCOORD", 3, 4, 0, 1),
            element("VICTORY_VERTEX_1_TEXCOORD", 4, 4, 0, 1),
            element("VICTORY_VERTEX_2_TEXCOORD", 5, 4, 0, 1),
        ];
        ps.outputs = vec![element("SV_Target", 0, 4, 16, 0)];
        let load = |s: &mut Shader, input, column| {
            let args = [
                s.m.i32(input),
                s.m.i32(0),
                s.m.i8(column),
                s.m.undef(s.t.i32),
            ];
            s.op(s.entry, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
        };
        let weights = [0, 1, 2].map(|column| load(&mut ps, 2, column));
        let mut interpolated = ps.m.f32(0);
        for vertex in 0..3 {
            let attribute = load(&mut ps, 3 + vertex as u32, 0);
            let term = ps.m.bin(ps.entry, Bin::Mul, attribute, weights[vertex]);
            interpolated = ps.m.bin(ps.entry, Bin::Add, interpolated, term);
        }
        let one = ps.m.f32(1f32.to_bits());
        for (column, value) in [weights[1], weights[2], interpolated, one]
            .into_iter()
            .enumerate()
        {
            store(&mut ps, 0, column as u8, value);
        }
        ps.m.terminate(ps.entry, Term::Ret);
        (
            wrap(crate::native::HwStage::Vs, vs),
            wrap(crate::native::HwStage::Ps, ps),
        )
    }

    #[test]
    fn software_barycentrics_render_both_halves_of_a_rectangle() {
        let Ok(host) = Host::new(false) else {
            eprintln!("no hardware device; not run");
            return;
        };
        let (vs, ps) = shaders(2, host.profile());
        let gs = super::super::rect_list_gs(&vs.dxil, Some(&ps.dxil)).unwrap();
        let root = host.root_signature(&vs, &ps).unwrap();
        let pipeline = host
            .pipeline_with_gs(&root, &vs.dxil, Some(&gs), &ps.dxil)
            .unwrap();
        let vertices = [
            -1., 1., 0.5, 1., 5., 0., 0., 1., -1., -1., 0.5, 1., 9., 0., 0., 1., 1., -1., 0.5, 1.,
            13., 0., 0., 1.,
        ];
        let pixels = host
            .draw(
                &root,
                &pipeline,
                &Draw {
                    roots: vec![],
                    aliases: &[],
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 32,
                    size: (8, 8),
                    mesh_groups: 0,
                },
            )
            .unwrap();
        for (i, pixel) in pixels.iter().enumerate() {
            let expected = 5. + ((i % 8 + i / 8) as f32 + 1.) / 2.;
            assert!(
                (pixel[2] - expected).abs() < 0.001 && pixel[3] == 1.,
                "texel {i}: {pixel:?}, expected {expected}"
            );
        }
    }

    #[test]
    fn software_barycentrics_render_perspective_and_linear_weights_on_gpu() {
        let Ok(host) = Host::new(false) else {
            eprintln!("no hardware device; not run");
            return;
        };
        for mode in [2, 4] {
            let (vs, ps) = shaders(mode, host.profile());
            let gs = barycentric_gs(&vs.dxil, &ps.dxil).unwrap().unwrap();
            if let Ok(mut validator) = super::super::validator::Validator::load(
                super::super::validator::Validator::default_path().as_deref(),
            ) {
                for code in [&vs.dxil, &ps.dxil, &gs] {
                    validator.validate(&mut code.clone()).unwrap();
                }
            }
            let root = host.root_signature(&vs, &ps).unwrap();
            let pipeline = host
                .pipeline_with_gs(&root, &vs.dxil, Some(&gs), &ps.dxil)
                .unwrap();
            let vertices = [
                -1., 1., 0.5, 1., 5., 0., 0., 1., -2., -6., 1., 2., 9., 0., 0., 1., 9., 3., 1.5,
                3., 13., 0., 0., 1.,
            ];
            let pixels = host
                .draw(
                    &root,
                    &pipeline,
                    &Draw {
                        roots: vec![],
                        aliases: &[],
                        textures: &[],
                        samplers: &[],
                        vertices: &vertices,
                        stride: 32,
                        size: (8, 8),
                        mesh_groups: 0,
                    },
                )
                .unwrap();
            for (i, pixel) in pixels.iter().enumerate() {
                let b = (i / 8) as f32 / 16. + 0.5 / 16.;
                let c = (i % 8) as f32 / 16. + 0.5 / 16.;
                let mut weights = [1. - b - c, b, c];
                if mode == 2 {
                    weights[1] /= 2.;
                    weights[2] /= 3.;
                    let sum = weights.iter().sum::<f32>();
                    weights.iter_mut().for_each(|weight| *weight /= sum);
                }
                let expected = [
                    weights[1],
                    weights[2],
                    weights[0] * 5. + weights[1] * 9. + weights[2] * 13.,
                    1.,
                ];
                assert!(
                    pixel
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| (*actual - expected).abs() < 0.001),
                    "mode {mode}, texel {i}: {pixel:?}, expected {expected:?}"
                );
            }
        }
    }
}
