use std::collections::{BTreeMap, BTreeSet};

use super::super::shader::Shader;
use super::Op;

pub(crate) fn compact_software_barycentrics(shader: &mut Shader) -> Result<(), &'static str> {
    if !shader.pixel
        || !shader.inputs.iter().any(|element| {
            element.name == "VICTORY_BARYCENTRICS" || element.name.starts_with("VICTORY_VERTEX_")
        })
    {
        return Ok(());
    }

    let mut keep: Vec<bool> = shader
        .inputs
        .iter()
        .map(|element| element.usage != 0 || matches!(element.kind, 3..=7))
        .collect();
    let mut calls = Vec::new();
    let emitted = shader.m.live();
    for (index, inst) in shader.m.insts.iter().enumerate() {
        if !emitted[index] {
            continue;
        }
        let Op::Call(function, args) = &inst.op else {
            continue;
        };
        let name = &shader.m.funcs[*function as usize].name;
        if !name.starts_with("dx.op.loadInput.") && !name.starts_with("dx.op.attributeAtVertex.") {
            continue;
        }
        let Some(&signature) = args.get(1) else {
            return Err("a software barycentric input call without a signature ID");
        };
        let Some(id) = shader
            .m
            .int_value(signature)
            .and_then(|id| usize::try_from(id).ok())
        else {
            return Err("a computed software barycentric input signature ID");
        };
        let Some(used) = keep.get_mut(id) else {
            return Err("a software barycentric input signature ID outside its signature");
        };
        *used = true;
        calls.push((index, id));
    }

    let rows: BTreeSet<u32> = shader
        .inputs
        .iter()
        .zip(&keep)
        .filter(|(_, keep)| **keep)
        .filter_map(|(element, _)| element.row)
        .collect();
    if rows.len() > 32 {
        return Err("software barycentrics exceeding 32 signature rows");
    }
    let rows: BTreeMap<u32, u32> = rows
        .into_iter()
        .enumerate()
        .map(|(new, old)| (old, new as u32))
        .collect();
    let mut ids = vec![None; keep.len()];
    let mut inputs = Vec::new();
    for (old, (element, keep)) in shader.inputs.iter().zip(keep).enumerate() {
        if !keep {
            continue;
        }
        let mut element = element.clone();
        element.row = element.row.map(|row| rows[&row]);
        ids[old] = Some(inputs.len() as u32);
        inputs.push(element);
    }

    for (index, old) in calls {
        let id = shader
            .m
            .i32(ids[old].expect("a referenced input is retained"));
        let Op::Call(_, args) = &mut shader.m.insts[index].op else {
            unreachable!()
        };
        args[1] = id;
    }
    shader.inputs = inputs;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::ShaderProfile;
    use crate::emit::bitcode::{Attr, Bin, V};
    use crate::emit::container::SigElement;

    fn element(name: &str, row: u32, col: u8, cols: u8, usage: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 1,
            row: Some(row),
            col,
            cols,
            usage,
            half: false,
        }
    }

    fn shader() -> Shader {
        let mut shader = Shader::new(true, ShaderProfile::Sm65);
        shader.inputs = vec![
            element("UNUSED", 0, 0, 4, 0),
            element("VICTORY_VERTEX_0_TEXCOORD", 1, 0, 2, 3),
            element("COLOR", 1, 2, 2, 3),
            element("UNUSED_GAP", 17, 0, 4, 0),
            element("VICTORY_VERTEX_1_TEXCOORD", 40, 0, 2, 3),
        ];
        shader
    }

    fn call(shader: &mut Shader, name: &str, id: V, vertex: V) -> V {
        let args = [id, shader.m.i32(0), shader.m.i8(1), vertex];
        shader.op(
            shader.entry,
            name,
            Attr::ReadNone,
            shader.t.f32,
            if name.starts_with("attributeAtVertex.") {
                137
            } else {
                4
            },
            &args,
        )
    }

    fn argument(shader: &Shader, value: V, argument: usize) -> V {
        let V::Inst(index) = value else {
            panic!("instruction expected")
        };
        let Op::Call(_, args) = &shader.m.insts[index as usize].op else {
            panic!("call expected")
        };
        args[argument]
    }

    #[test]
    fn compact_barycentric_ids_preserve_shared_constants_and_packed_columns() {
        let mut shader = shader();
        let old = shader.m.i32(4);
        let two = shader.m.i32(2);
        let arithmetic = shader.m.bin(shader.entry, Bin::Add, old, two);
        let vertex = shader.m.i32(0);
        let input = call(&mut shader, "loadInput.f32", old, vertex);
        compact_software_barycentrics(&mut shader).unwrap();
        assert_eq!(shader.inputs.len(), 3);
        assert_eq!(shader.inputs[0].row, Some(0));
        assert_eq!(shader.inputs[1].row, Some(0));
        assert_eq!((shader.inputs[1].col, shader.inputs[1].cols), (2, 2));
        assert_eq!(shader.inputs[2].row, Some(1));
        assert_eq!(shader.m.int_value(argument(&shader, input, 1)), Some(2));
        assert_eq!(shader.m.int_value(old), Some(4));
        let V::Inst(index) = arithmetic else { panic!() };
        assert!(
            matches!(shader.m.insts[index as usize].op, Op::Bin(Bin::Add, a, b) if a == old && b == two)
        );
    }

    #[test]
    fn compact_barycentric_inputs_ignore_unemitted_arena_calls_after_dead_cleanup() {
        let mut shader = shader();
        let mut dead_calls = Vec::new();
        for row in 100..140 {
            let id = shader.inputs.len() as u32;
            shader
                .inputs
                .push(element(&format!("UNUSED_{row}"), row, 0, 4, 0));
            let id = shader.m.i32(id);
            let vertex = shader.m.i32(0);
            dead_calls.push(call(&mut shader, "loadInput.f32", id, vertex));
        }
        let id = shader.m.i32(4);
        let vertex = shader.m.i32(0);
        let kept = call(&mut shader, "loadInput.f32", id, vertex);
        let args = [shader.m.i32(0), shader.m.i32(0), shader.m.i8(0), kept];
        shader.op(
            shader.entry,
            "storeOutput.f32",
            Attr::NoUnwind,
            shader.t.void,
            5,
            &args,
        );
        shader
            .m
            .terminate(shader.entry, crate::emit::bitcode::Term::Ret);
        shader.m.optimise(crate::emit::bitcode::opt::DEAD);
        let emitted = shader.m.live();
        for &call in &dead_calls {
            let V::Inst(index) = call else { panic!() };
            assert!(matches!(shader.m.insts[index as usize].op, Op::Call(..)));
            assert!(!emitted[index as usize]);
        }
        compact_software_barycentrics(&mut shader).unwrap();
        assert_eq!(shader.inputs.len(), 3);
        assert_eq!(shader.m.int_value(argument(&shader, kept, 1)), Some(2));
        for call in dead_calls {
            assert!(shader.m.int_value(argument(&shader, call, 1)).unwrap() >= 5);
        }
    }
    #[test]
    fn compact_barycentric_inputs_preserve_dynamic_vertex_selection() {
        let mut shader = shader();
        let id = shader.m.i32(4);
        let vertex = shader.op(
            shader.entry,
            "primitiveID.i32",
            Attr::ReadNone,
            shader.t.i32,
            108,
            &[],
        );
        let i8t = shader.m.int(8);
        let vertex = shader
            .m
            .cast(shader.entry, crate::emit::bitcode::Cast::Trunc, vertex, i8t);
        let input = call(&mut shader, "attributeAtVertex.f32", id, vertex);
        compact_software_barycentrics(&mut shader).unwrap();
        assert_eq!(shader.m.int_value(argument(&shader, input, 1)), Some(2));
        assert_eq!(argument(&shader, input, 4), vertex);
    }

    #[test]
    fn compact_barycentric_inputs_reject_unknown_signature_ids_without_mutation() {
        let mut shader = shader();
        let old_inputs = shader.inputs.clone();
        let unknown = shader.op(
            shader.entry,
            "primitiveID.i32",
            Attr::ReadNone,
            shader.t.i32,
            108,
            &[],
        );
        let vertex = shader.m.i32(0);
        call(&mut shader, "loadInput.f32", unknown, vertex);
        assert_eq!(
            compact_software_barycentrics(&mut shader),
            Err("a computed software barycentric input signature ID")
        );
        assert_eq!(shader.inputs, old_inputs);
    }

    #[test]
    fn compact_barycentric_inputs_keep_referenced_and_raster_system_elements() {
        let mut shader = shader();
        shader.inputs[0].name = "SV_Position".into();
        shader.inputs[0].kind = 3;
        shader.inputs[4].usage = 0;
        let id = shader.m.i32(4);
        let vertex = shader.m.i32(0);
        let input = call(&mut shader, "loadInput.f32", id, vertex);
        compact_software_barycentrics(&mut shader).unwrap();
        assert_eq!(shader.inputs[0].kind, 3);
        assert_eq!(shader.inputs.len(), 4);
        assert_eq!(shader.m.int_value(argument(&shader, input, 1)), Some(3));
    }

    #[test]
    fn compact_barycentric_inputs_remove_unused_front_face_but_keep_used_system_values() {
        let mut shader = shader();
        let mut face = element("SV_IsFrontFace", 50, 0, 1, 0);
        face.kind = 13;
        face.comp = 5;
        shader.inputs.push(face);
        let mut primitive = element("SV_PrimitiveID", 51, 0, 1, 1);
        primitive.kind = 10;
        primitive.comp = 5;
        shader.inputs.push(primitive);
        compact_software_barycentrics(&mut shader).unwrap();
        assert_eq!(shader.inputs.len(), 4);
        assert!(!shader.inputs.iter().any(|element| element.kind == 13));
        assert_eq!(shader.inputs.last().unwrap().kind, 10);
        assert_eq!(shader.inputs.last().unwrap().row, Some(2));
    }
    #[test]
    fn compact_barycentric_inputs_refuse_more_than_32_required_rows() {
        let mut shader = shader();
        shader.inputs = (0..33)
            .map(|row| element(&format!("VICTORY_VERTEX_0_TEXCOORD_{row}"), row, 0, 4, 15))
            .collect();
        let before = shader.inputs.clone();
        assert_eq!(
            compact_software_barycentrics(&mut shader),
            Err("software barycentrics exceeding 32 signature rows")
        );
        assert_eq!(shader.inputs, before);
    }

    #[test]
    fn compact_barycentric_inputs_do_not_touch_other_shader_signatures() {
        let mut shader = shader();
        shader.pixel = false;
        let before = shader.inputs.clone();
        compact_software_barycentrics(&mut shader).unwrap();
        assert_eq!(shader.inputs, before);
    }
}
