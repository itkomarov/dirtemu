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

fn passthrough_outputs(
    s: &mut Shader,
    wanted: &[SigElement],
) -> Result<Vec<(usize, usize)>, ShaderError> {
    let mut occupied: std::collections::BTreeSet<u32> =
        wanted.iter().filter_map(|element| element.row).collect();
    if occupied.iter().any(|row| *row >= 32) {
        return Err(unsupported(
            "software barycentrics exceeding 32 signature rows",
        ));
    }
    let mut extra_rows = std::collections::BTreeMap::new();
    let mut passthrough = Vec::new();
    for input in 0..s.inputs.len() {
        let source = &s.inputs[input];
        let matching = wanted
            .iter()
            .find(|element| element.name == source.name && element.index == source.index);
        let raster = matches!(source.kind, 3..=7);
        if matching.is_none() && !raster {
            continue;
        }
        let mut output = source.clone();
        if matching
            .is_some_and(|matching| matching.cols > output.cols || matching.comp != output.comp)
        {
            return Err(unsupported(
                "a pixel input differing from its vertex output",
            ));
        }
        if let Some(matching) = matching.filter(|matching| !raster || matching.cols == output.cols)
        {
            output.row = matching.row;
            output.col = matching.col;
            output.cols = matching.cols;
            output.usage = (1 << output.cols) - 1;
        } else {
            let key = output
                .row
                .ok_or_else(|| unsupported("a raster output without a register"))?;
            let row = if let Some(&row) = extra_rows.get(&key) {
                row
            } else {
                let row = (0..32).find(|row| !occupied.contains(row)).ok_or_else(|| {
                    unsupported("software barycentrics exceeding 32 signature rows")
                })?;
                occupied.insert(row);
                extra_rows.insert(key, row);
                row
            };
            output.row = Some(row);
        }
        let id = s.outputs.len();
        s.outputs.push(output);
        passthrough.push((input, id));
    }
    Ok(passthrough)
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
    s.inputs = outputs;
    let passthrough = passthrough_outputs(&mut s, wanted)?;
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
        for &(input, output) in &passthrough {
            for column in 0..s.outputs[output].cols {
                let value = inputs[input][column as usize][vertex];
                store(&mut s, output, column, value);
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
            profile: ShaderProfile::Sm65,
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

    fn shader_sources(
        mode: u8,
        padding: u32,
        dynamic_vertex: bool,
        primitive_id: bool,
    ) -> (Shader, Shader) {
        let mut vs = Shader::new(false, ShaderProfile::Sm65);
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
        let mut ps = Shader::new(true, ShaderProfile::Sm65);
        ps.inputs = vec![
            element("SV_Position", 0, 4, 3, 4),
            element("TEXCOORD", 1, 4, 0, 1),
            element("VICTORY_BARYCENTRICS", 2, 3, 0, mode),
            element("VICTORY_VERTEX_0_TEXCOORD", 3, 4, 0, 1),
            element("VICTORY_VERTEX_1_TEXCOORD", 4, 4, 0, 1),
            element("VICTORY_VERTEX_2_TEXCOORD", 5, 4, 0, 1),
        ];
        ps.inputs[1].usage = 0;
        let mut aliases = ps.inputs.split_off(2);
        for row in 0..padding {
            let mut unused = element(&format!("UNUSED_PIXEL_{row}"), row + 2, 4, 0, 1);
            unused.usage = 0;
            ps.inputs.push(unused);
        }
        for alias in &mut aliases {
            alias.row = alias.row.map(|row| row + padding);
        }
        ps.inputs.extend(aliases);
        let primitive_input = primitive_id.then(|| {
            let mut primitive = element("SV_PrimitiveID", 6 + padding, 1, 10, 1);
            primitive.comp = 5;
            let id = ps.inputs.len() as u32;
            ps.inputs.push(primitive);
            id
        });
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
        let weights = [0, 1, 2].map(|column| load(&mut ps, 2 + padding, column));
        let mut interpolated = ps.m.f32(0);
        for vertex in 0..3 {
            let attribute = load(&mut ps, 3 + padding + vertex as u32, 0);
            let term = ps.m.bin(ps.entry, Bin::Mul, attribute, weights[vertex]);
            interpolated = ps.m.bin(ps.entry, Bin::Add, interpolated, term);
        }
        if dynamic_vertex {
            let mut largest = weights[0];
            let mut selected = load(&mut ps, 3 + padding, 0);
            for vertex in 1..3 {
                let wins = ps.m.cmp(
                    ps.entry,
                    super::super::bitcode::pred::OGT,
                    weights[vertex],
                    largest,
                );
                let attribute = load(&mut ps, 3 + padding + vertex as u32, 0);
                selected = ps.m.select(ps.entry, wins, attribute, selected);
                largest = ps.m.select(ps.entry, wins, weights[vertex], largest);
            }
            interpolated = selected;
        }
        let one = ps.m.f32(1f32.to_bits());
        let alpha = if let Some(input) = primitive_input {
            let args = [
                ps.m.i32(input),
                ps.m.i32(0),
                ps.m.i8(0),
                ps.m.undef(ps.t.i32),
            ];
            let id = ps.op(
                ps.entry,
                "loadInput.i32",
                Attr::ReadNone,
                ps.t.i32,
                4,
                &args,
            );
            let id =
                ps.m.cast(ps.entry, super::super::bitcode::Cast::UiToFp, id, ps.t.f32);
            ps.m.bin(ps.entry, Bin::Add, id, one)
        } else {
            one
        };
        for (column, value) in [weights[1], weights[2], interpolated, alpha]
            .into_iter()
            .enumerate()
        {
            store(&mut ps, 0, column as u8, value);
        }
        ps.m.terminate(ps.entry, Term::Ret);
        (vs, ps)
    }

    fn shaders(mode: u8) -> (HostShader, HostShader) {
        let (vs, ps) = shader_sources(mode, 0, false, false);
        (
            wrap(crate::native::HwStage::Vs, vs),
            wrap(crate::native::HwStage::Ps, ps),
        )
    }

    #[test]
    fn software_barycentrics_render_both_halves_of_a_rectangle() {
        let host = Host::new(false).expect("hardware SM 6.5 device");
        let (vs, ps) = shaders(2);
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
        let host = Host::new(false).expect("hardware SM 6.5 device");
        for mode in [2, 4] {
            let (vs, ps) = shaders(mode);
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
    #[test]
    fn software_barycentrics_compact_wide_inputs_and_keep_dynamic_vertices_and_primitive_ids() {
        let host = Host::new(false).expect("hardware SM 6.5 device");
        let (vs, ps) = shader_sources(4, 35, true, true);
        assert!(
            ps.inputs
                .iter()
                .filter_map(|element| element.row)
                .max()
                .unwrap()
                >= 40
        );
        let vertex = wrap(crate::native::HwStage::Vs, vs);
        let pixel = wrap(crate::native::HwStage::Ps, ps);
        let pixel_container = Container::parse(&pixel.dxil).unwrap();
        let inputs =
            super::super::signature(pixel_container.part(FourCC::ISG1).unwrap().data).unwrap();
        assert_eq!(inputs.len(), 6);
        assert_eq!(
            inputs.iter().filter_map(|element| element.row).max(),
            Some(5)
        );
        let gs = barycentric_gs(&vertex.dxil, &pixel.dxil).unwrap().unwrap();
        let gs_container = Container::parse(&gs).unwrap();
        let outputs =
            super::super::signature(gs_container.part(FourCC::OSG1).unwrap().data).unwrap();
        assert_eq!(outputs.len(), 6);
        assert!(!outputs.iter().any(|element| element.name == "UNUSED"));
        let mut validator = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        )
        .expect("Microsoft DXIL validator");
        for code in [&vertex.dxil, &pixel.dxil, &gs] {
            validator.validate(&mut code.clone()).unwrap();
        }
        let root = host.root_signature(&vertex, &pixel).unwrap();
        let pipeline = host
            .pipeline_with_gs(&root, &vertex.dxil, Some(&gs), &pixel.dxil)
            .unwrap();
        let vertices = [
            -1., 1., 0.5, 1., 5., 0., 0., 1., -1., -1., 0.5, 1., 9., 0., 0., 1., 1., -1., 0.5, 1.,
            13., 0., 0., 1., -1., 1., 0.5, 1., 21., 0., 0., 1., 1., -1., 0.5, 1., 25., 0., 0., 1.,
            1., 1., 0.5, 1., 29., 0., 0., 1.,
        ];
        let pixels = host
            .draw_vertices(
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
                6,
            )
            .unwrap();
        let mut seen = [[false; 3]; 2];
        for (index, pixel) in pixels.iter().enumerate() {
            let (x, y) = (index % 8, index / 8);
            if x == y {
                continue;
            }
            let (u, v) = ((x as f32 + 0.5) / 8., (y as f32 + 0.5) / 8.);
            let (primitive, weights, values) = if x < y {
                (0, [1. - v, v - u, u], [5., 9., 13.])
            } else {
                (1, [1. - u, v, u - v], [21., 25., 29.])
            };
            let maximum = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if weights
                .iter()
                .filter(|weight| (**weight - maximum).abs() < 0.000001)
                .count()
                > 1
            {
                continue;
            }
            let mut selected = 0;
            for vertex in 1..3 {
                if weights[vertex] > weights[selected] {
                    selected = vertex;
                }
            }
            seen[primitive][selected] = true;
            let expected = [
                weights[1],
                weights[2],
                values[selected],
                primitive as f32 + 1.,
            ];
            assert!(
                pixel
                    .iter()
                    .zip(expected)
                    .all(|(got, wanted)| (*got - wanted).abs() < 0.001),
                "texel {x},{y}: {pixel:?}, expected {expected:?}"
            );
        }
        assert!(
            seen.iter().flatten().all(|seen| *seen),
            "all three source vertices of both primitives must be observed: {seen:?}"
        );
    }
}
#[cfg(test)]
mod layout_tests {
    use super::super::ShaderProfile;
    use super::*;

    fn element(name: &str, index: u32, row: u32, col: u8, cols: u8, kind: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp: 9,
            interp: if kind == 3 { 4 } else { 1 },
            row: Some(row),
            col,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    #[test]
    fn barycentric_passthrough_drops_unused_outputs_but_keeps_alias_sources() {
        let mut shader = Shader::new(false, ShaderProfile::Sm65);
        shader.inputs = vec![
            element("SV_Position", 0, 0, 0, 4, 3),
            element("TEXCOORD", 0, 1, 0, 4, 0),
        ];
        shader
            .inputs
            .extend((2..40).map(|row| element("UNUSED", row, row, 0, 4, 0)));
        let mut wanted = vec![element("SV_Position", 0, 0, 0, 4, 3)];
        for vertex in 0..3 {
            wanted.push(element(
                &format!("VICTORY_VERTEX_{vertex}_TEXCOORD"),
                0,
                vertex + 1,
                0,
                4,
                0,
            ));
        }
        wanted.push(element("VICTORY_BARYCENTRICS", 0, 4, 0, 3, 0));
        let passthrough = passthrough_outputs(&mut shader, &wanted).unwrap();
        let aliases = append(&mut shader, &wanted).unwrap();
        assert_eq!(passthrough, [(0, 0)]);
        assert_eq!(shader.inputs.len(), 40);
        assert_eq!(shader.outputs.len(), 5);
        assert_eq!(
            aliases
                .iter()
                .filter_map(|alias| alias.source)
                .collect::<Vec<_>>(),
            [(1, 0), (1, 1), (1, 2)]
        );
        assert!(
            shader
                .outputs
                .iter()
                .all(|element| element.row.unwrap() < 5)
        );
    }

    #[test]
    fn barycentric_passthrough_remaps_packed_semantics_without_changing_input_ids() {
        let mut shader = Shader::new(false, ShaderProfile::Sm65);
        shader.inputs = vec![
            element("SV_Position", 0, 0, 0, 4, 3),
            element("UNUSED", 0, 1, 0, 4, 0),
            element("COLOR", 3, 9, 0, 2, 0),
            element("TEXCOORD", 2, 9, 2, 2, 0),
        ];
        let wanted = vec![
            element("SV_Position", 0, 0, 0, 4, 3),
            element("COLOR", 3, 1, 0, 2, 0),
            element("TEXCOORD", 2, 1, 2, 2, 0),
        ];
        let passthrough = passthrough_outputs(&mut shader, &wanted).unwrap();
        assert_eq!(passthrough, [(0, 0), (2, 1), (3, 2)]);
        assert_eq!(shader.inputs[2].row, Some(9));
        assert_eq!((shader.outputs[1].row, shader.outputs[1].col), (Some(1), 0));
        assert_eq!((shader.outputs[2].row, shader.outputs[2].col), (Some(1), 2));
    }

    #[test]
    fn barycentric_passthrough_keeps_raster_outputs_when_the_pixel_does_not_read_them() {
        let mut shader = Shader::new(false, ShaderProfile::Sm65);
        shader.inputs = vec![
            element("SV_Position", 0, 0, 0, 4, 3),
            element("SV_ClipDistance", 0, 1, 0, 2, 6),
            element("SV_CullDistance", 0, 1, 2, 2, 7),
            element("SV_RenderTargetArrayIndex", 0, 2, 0, 1, 4),
            element("SV_ViewportArrayIndex", 0, 3, 0, 1, 5),
            element("UNUSED", 0, 4, 0, 4, 0),
        ];
        let wanted = [element("VICTORY_BARYCENTRICS", 0, 0, 0, 3, 0)];
        let passthrough = passthrough_outputs(&mut shader, &wanted).unwrap();
        assert_eq!(passthrough.len(), 5);
        assert_eq!(
            shader
                .outputs
                .iter()
                .map(|element| element.kind)
                .collect::<Vec<_>>(),
            [3, 6, 7, 4, 5]
        );
        assert!(shader.outputs.iter().all(|element| element.row != Some(0)));
        assert_eq!(shader.outputs[1].row, shader.outputs[2].row);
    }

    #[test]
    fn barycentric_passthrough_accepts_an_ordinary_input_that_reads_fewer_components() {
        let mut shader = Shader::new(false, ShaderProfile::Sm65);
        shader.inputs = vec![element("COLOR", 0, 5, 0, 4, 0)];
        let wanted = [element("COLOR", 0, 1, 2, 2, 0)];
        assert_eq!(passthrough_outputs(&mut shader, &wanted).unwrap(), [(0, 0)]);
        assert_eq!(shader.inputs[0].cols, 4);
        assert_eq!(
            (
                shader.outputs[0].row,
                shader.outputs[0].col,
                shader.outputs[0].cols
            ),
            (Some(1), 2, 2)
        );
    }

    #[test]
    fn barycentric_passthrough_never_truncates_clipping_for_a_smaller_pixel_input() {
        let mut shader = Shader::new(false, ShaderProfile::Sm65);
        shader.inputs = vec![element("SV_ClipDistance", 0, 5, 0, 2, 6)];
        let wanted = [element("SV_ClipDistance", 0, 0, 0, 1, 6)];
        assert_eq!(passthrough_outputs(&mut shader, &wanted).unwrap(), [(0, 0)]);
        assert_eq!(shader.outputs[0].cols, 2);
        assert_ne!(shader.outputs[0].row, Some(0));
    }
    #[test]
    fn barycentric_passthrough_never_drops_position_to_force_a_full_signature_to_fit() {
        let mut shader = Shader::new(false, ShaderProfile::Sm65);
        shader.inputs = vec![element("SV_Position", 0, 0, 0, 4, 3)];
        let wanted: Vec<_> = (0..32)
            .map(|row| element("VICTORY_BARYCENTRICS", row, row, 0, 4, 0))
            .collect();
        assert!(passthrough_outputs(&mut shader, &wanted).is_err());
    }
}
