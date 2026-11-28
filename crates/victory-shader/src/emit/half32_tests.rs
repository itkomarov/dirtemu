use super::bitcode::{Attr, Bin, Term, V, pred};
use super::host::{Dispatch, Root, View};
use super::shader::{DescriptorAlias, Shader};
use super::{HostShader, ShaderProfile, UserData, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

fn store(s: &mut Shader, block: usize, handle: V, address: V, value: V) {
    let undef = s.m.undef(s.t.i32);
    let mask = s.m.i8(1);
    let align = s.m.i32(4);
    s.op(
        block,
        "rawBufferStore.i32",
        Attr::NoUnwind,
        s.t.void,
        140,
        &[
            handle, address, undef, value, undef, undef, undef, mask, align,
        ],
    );
}

fn finish(s: Shader, user_data: Vec<UserData>) -> HostShader {
    assert!(!s.half32_unsupported);
    let descriptor_aliases = s
        .descriptor_tables
        .iter()
        .map(|t| DescriptorAlias {
            class: t.key.class,
            space: t.space,
        })
        .collect();
    let profile = s.profile;
    let direct_heap = s.flags & super::shader::flags::RESOURCE_HEAP != 0;
    HostShader {
        stage: HwStage::Cs,
        profile,
        descriptor_aliases,
        direct_heap,
        dxil: s.try_finish().unwrap(),
        mesh_fallback: None,
        user_data,
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    }
}

fn primitive_shader(profile: ShaderProfile) -> HostShader {
    let mut s = Shader::new(false, profile);
    s.threads = Some([256, 1, 1]);
    s.wave_size = Some(32);
    s.half32 = true;
    s.flags |= super::shader::flags::WAVE_OPS;
    s.user_data = 4;
    s.lds_dwords = 256;
    let b = s.entry;
    let i32t = s.t.i32;
    let zero = s.m.i32(0);
    let user = s.user_data_handle();
    let constants = s.op(
        b,
        "cbufferLoadLegacy.i32",
        Attr::ReadOnly,
        s.t.cbuf_ret,
        59,
        &[user, zero],
    );
    let values = [0, 1, 2, 3].map(|i| s.m.extract(b, constants, i));
    let flat = s.op(
        b,
        "flattenedThreadIdInGroup.i32",
        Attr::ReadNone,
        i32t,
        96,
        &[],
    );
    let group = s.op(b, "groupId.i32", Attr::ReadNone, i32t, 94, &[zero]);
    let lane = s.op(b, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
    let k32 = s.m.i32(32);
    let half = s.m.bin(b, Bin::And, flat, k32);
    let upper = s.m.cmp(b, pred::NE, half, zero);
    let predicate = s.m.select(b, upper, values[1], values[0]);
    let active = s.m.select(b, upper, values[3], values[2]);
    let one = s.m.i32(1);
    let bit = s.m.bin(b, Bin::Shl, one, lane);
    let predicate = s.m.bin(b, Bin::And, predicate, bit);
    let predicate = s.m.cmp(b, pred::NE, predicate, zero);
    let active = s.m.bin(b, Bin::And, active, bit);
    let active = s.m.cmp(b, pred::NE, active, zero);
    let n = s.m.i32(256);
    let base = s.m.bin(b, Bin::Mul, group, n);
    let global = s.m.bin(b, Bin::Add, base, flat);
    let seed = s.m.i32(0x1000);
    let payload = s.m.bin(b, Bin::Add, global, seed);
    let last = s.m.i32(31);
    let read31 = s.op(
        b,
        "waveReadLaneAt.i32",
        Attr::NoUnwind,
        i32t,
        117,
        &[payload, last],
    );
    let lds = s.lds();
    let address = s.m.element(b, lds, flat);
    s.m.store(b, address, payload);
    let sync = s.m.i32(9);
    s.op(b, "barrier", Attr::NoUnwind, s.t.void, 80, &[sync]);
    let mask = s.m.i32(!31);
    let wave_start = s.m.bin(b, Bin::And, flat, mask);
    let address = s.m.element(b, lds, wave_start);
    let lds_first = s.m.load(b, address);
    let handle = s.heap_handle(b, zero, false, false, (11 | 0x1000, 0));
    let stride = s.m.i32(40);
    let address = s.m.bin(b, Bin::Mul, global, stride);
    for (slot, value) in [(0, flat), (1, lane), (2, read31), (9, lds_first)] {
        let off = s.m.i32(slot * 4);
        let at = s.m.bin(b, Bin::Add, address, off);
        store(&mut s, b, handle, at, value);
    }
    for slot in 3..9 {
        let off = s.m.i32(slot * 4);
        let at = s.m.bin(b, Bin::Add, address, off);
        store(&mut s, b, handle, at, zero);
    }
    let inside = s.m.new_block();
    let after = s.m.new_block();
    s.m.terminate(b, Term::CondBr(active, inside, after));
    let first = s.op(
        inside,
        "waveReadLaneFirst.i32",
        Attr::NoUnwind,
        i32t,
        118,
        &[payload],
    );
    let four = s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
    let ballot = s.op(
        inside,
        "waveActiveBallot",
        Attr::NoUnwind,
        four,
        116,
        &[predicate],
    );
    let words = [0, 1, 2, 3].map(|i| s.m.extract(inside, ballot, i));
    let count = s.op(
        inside,
        "unaryBits.i32",
        Attr::ReadNone,
        i32t,
        31,
        &[words[0]],
    );
    for (i, value) in [first, words[0], words[1], words[2], words[3], count]
        .into_iter()
        .enumerate()
    {
        let off = s.m.i32((i as u32 + 3) * 4);
        let at = s.m.bin(inside, Bin::Add, address, off);
        store(&mut s, inside, handle, at, value);
    }
    s.m.terminate(inside, Term::Br(after));
    s.m.terminate(after, Term::Ret);
    finish(
        s,
        (0..4)
            .map(|index| UserData {
                index,
                slot: 0,
                root: RootType::Constants,
                dword: index,
            })
            .collect(),
    )
}

pub(super) fn host_and_settings() -> (super::host::Host, super::lower::tests::RestoreHostMinWave) {
    let (host, settings) =
        super::lower::tests::shader_regression_host().expect("half32 regression device");
    assert_eq!(
        lower::host_min_wave(),
        64,
        "requires a physical wave64 device"
    );
    assert_eq!(
        lower::host_max_wave(),
        64,
        "requires a physical wave64 device"
    );
    (host, settings)
}

#[test]
fn half32_primitives_keep_two_masks_and_first_lanes_independent() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if std::env::var_os("VICTORY_TEST_GPU_HARDWARE").is_none() {
        eprintln!("SKIP half32 numeric requires VICTORY_TEST_GPU_HARDWARE=1");
        return;
    }
    let (host, _settings) = host_and_settings();
    let profile = super::resolved_shader_profile().unwrap().profile;
    let shader = primitive_shader(profile);
    let mut validator =
        validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut shader.dxil.clone()).unwrap();
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let aliases = host.compute_alias_parameters(&shader);
    for constants in [
        [0, 0, u32::MAX, u32::MAX],
        [1, 0, u32::MAX, u32::MAX],
        [2, 0, u32::MAX, u32::MAX],
        [1 << 31, 0, u32::MAX, u32::MAX],
        [0, 1, u32::MAX, u32::MAX],
        [0, 2, u32::MAX, u32::MAX],
        [0, 1 << 31, u32::MAX, u32::MAX],
        [u32::MAX; 4],
        [u32::MAX, u32::MAX, 0, u32::MAX],
        [u32::MAX, u32::MAX, 0x8000_0002, 0x8000_0001],
        [u32::MAX, u32::MAX, 1, 1 << 31],
        [0; 4],
    ] {
        let got = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(constants.to_vec()))],
                    aliases: &aliases,
                    views: &[View::RawUav(3 * 256 * 10)],
                    samplers: &[],
                    groups: [3, 1, 1],
                },
            )
            .unwrap();
        let mut expected = Vec::new();
        for global in 0..3 * 256u32 {
            let flat = global % 256;
            let lane = flat & 31;
            let upper = (flat & 32 != 0) as usize;
            let active = constants[2 + upper];
            let mask = active & constants[upper];
            let enabled = active >> lane & 1 != 0;
            let wave_start = global & !31;
            expected.extend([
                flat,
                lane,
                wave_start + 31 + 0x1000,
                if enabled {
                    wave_start + active.trailing_zeros() + 0x1000
                } else {
                    0
                },
                if enabled { mask } else { 0 },
                0,
                0,
                0,
                if enabled { mask.count_ones() } else { 0 },
                wave_start + 0x1000,
            ]);
        }
        assert_eq!(got[0], expected, "constants={constants:08x?}");
    }
}

fn bit_reduction_shader(profile: ShaderProfile) -> HostShader {
    let mut s = Shader::new(false, profile);
    s.threads = Some([256, 1, 1]);
    s.wave_size = Some(32);
    s.half32 = true;
    s.flags |= super::shader::flags::WAVE_OPS;
    s.user_data = 4;
    let b = s.entry;
    let i32t = s.t.i32;
    let zero = s.m.i32(0);
    let user = s.user_data_handle();
    let constants = s.op(
        b,
        "cbufferLoadLegacy.i32",
        Attr::ReadOnly,
        s.t.cbuf_ret,
        59,
        &[user, zero],
    );
    let values = [0, 1, 2, 3].map(|i| s.m.extract(b, constants, i));
    let flat = s.op(
        b,
        "flattenedThreadIdInGroup.i32",
        Attr::ReadNone,
        i32t,
        96,
        &[],
    );
    let group = s.op(b, "groupId.i32", Attr::ReadNone, i32t, 94, &[zero]);
    let k31 = s.m.i32(31);
    let lane = s.m.bin(b, Bin::And, flat, k31);
    let k32 = s.m.i32(32);
    let half = s.m.bin(b, Bin::And, flat, k32);
    let upper = s.m.cmp(b, pred::NE, half, zero);
    let active = s.m.select(b, upper, values[1], values[0]);
    let seed = s.m.select(b, upper, values[3], values[2]);
    let one = s.m.i32(1);
    let bit = s.m.bin(b, Bin::Shl, one, lane);
    let payload = s.m.bin(b, Bin::Xor, bit, seed);
    let enabled = s.m.bin(b, Bin::And, active, bit);
    let enabled = s.m.cmp(b, pred::NE, enabled, zero);
    let n = s.m.i32(256);
    let group_start = s.m.bin(b, Bin::Mul, group, n);
    let global = s.m.bin(b, Bin::Add, group_start, flat);
    let stride = s.m.i32(12);
    let address = s.m.bin(b, Bin::Mul, global, stride);
    let handle = s.heap_handle(b, zero, false, false, (11 | 0x1000, 0));
    let sentinel = s.m.i32(0xCDCD_CDCD);
    for slot in 0..3 {
        let offset = s.m.i32(slot * 4);
        let at = s.m.bin(b, Bin::Add, address, offset);
        store(&mut s, b, handle, at, sentinel);
    }
    let inside = s.m.new_block();
    let after = s.m.new_block();
    s.m.terminate(b, Term::CondBr(enabled, inside, after));
    for kind in 0..3 {
        let op = s.m.i8(kind);
        let result = s.op(
            inside,
            "waveActiveBit.i32",
            Attr::NoUnwind,
            i32t,
            120,
            &[payload, op],
        );
        let offset = s.m.i32(u32::from(kind) * 4);
        let at = s.m.bin(inside, Bin::Add, address, offset);
        store(&mut s, inside, handle, at, result);
    }
    s.m.terminate(inside, Term::Br(after));
    s.m.terminate(after, Term::Ret);
    finish(
        s,
        (0..4)
            .map(|index| UserData {
                index,
                slot: 0,
                root: RootType::Constants,
                dword: index,
            })
            .collect(),
    )
}

#[test]
fn half32_bit_reductions_keep_each_active_half_independent() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if std::env::var_os("VICTORY_TEST_GPU_HARDWARE").is_none() {
        eprintln!("SKIP half32 bit reductions require VICTORY_TEST_GPU_HARDWARE=1");
        return;
    }
    let (host, _settings) = host_and_settings();
    let profile = super::resolved_shader_profile().unwrap().profile;
    let shader = bit_reduction_shader(profile);
    let mut validator =
        validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut shader.dxil.clone()).unwrap();
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let aliases = host.compute_alias_parameters(&shader);
    for masks in [
        [u32::MAX; 2],
        [0, u32::MAX],
        [u32::MAX, 0],
        [0, 0],
        [1, 1 << 31],
        [1 << 31, 1],
        [0xAAAA_AAAA, 0x5555_5555],
        [0x0000_FFFF, 0xFFFF_0000],
        [0x8000_0002, 0x4000_0001],
    ] {
        for seeds in [[0, 0], [u32::MAX, 0], [0xF0F0_FF00, 0x0F0F_00FF]] {
            let constants = [masks[0], masks[1], seeds[0], seeds[1]];
            let got = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(constants.to_vec()))],
                        aliases: &aliases,
                        views: &[View::RawUav(3 * 256 * 3)],
                        samplers: &[],
                        groups: [3, 1, 1],
                    },
                )
                .unwrap();
            let mut expected = Vec::new();
            for global in 0..3 * 256u32 {
                let lane = global & 31;
                let half = (global >> 5 & 1) as usize;
                if masks[half] >> lane & 1 == 0 {
                    expected.extend([0xCDCD_CDCD; 3]);
                    continue;
                }
                let mut reductions = [u32::MAX, 0, 0];
                for source in 0..32 {
                    if masks[half] >> source & 1 != 0 {
                        let value = (1 << source) ^ seeds[half];
                        reductions[0] &= value;
                        reductions[1] |= value;
                        reductions[2] ^= value;
                    }
                }
                expected.extend(reductions);
            }
            assert_eq!(got[0], expected, "constants={constants:08x?}");
        }
    }
}

fn vop2(op: u32, dst: u32, a: u32, b: u32) -> u32 {
    op << 25 | dst << 17 | b << 9 | a
}

fn native_threads(threads: [u32; 3]) -> NativeProgram {
    let count = threads.iter().product::<u32>();
    let mut words = vec![0xBE8C_0305, 0xBE85_0380, 0xF408_0202, 0xFA00_0000];
    words.extend([
        vop2(0x1a, 3, 128 + threads[0].ilog2(), 1),
        vop2(0x1a, 4, 128 + (threads[0] * threads[1]).ilog2(), 2),
        vop2(0x25, 3, 0x103, 0),
        vop2(0x25, 3, 0x103, 4),
        0x7E1C_0503,
        0xD760_000F,
        0x0001_3F03,
        vop2(0x1a, 11, 130, 3),
        0xD834_0000,
        0x0000_030B,
        0xBF8A_0000,
        0xD8D8_0000,
        0x0600_000B,
        0x7E0A_020C,
        vop2(0x1a, 5, 128 + count.ilog2(), 5),
        vop2(0x25, 5, 0x105, 3),
        vop2(0x1a, 10, 132, 5),
        0x7E0E_020E,
        0x7E10_020F,
        0x7E12_020C,
        0xE078_1000,
        0x8002_060A,
        0xBF81_0000,
    ]);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | (1 << 7) | (2 << 11) | ((count * 4).div_ceil(512) << 15),
            threads,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

pub(super) fn lower_native(
    native: &NativeProgram,
    profile: ShaderProfile,
) -> Result<lower::Lowered, ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    lower::lower_with_profile(
        &lower::Parts {
            native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            mesh_fallback: false,
        },
        profile,
    )
}

#[test]
fn half32_native_thread_ids_and_lds_remain_distinct_in_all_groups() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if std::env::var_os("VICTORY_TEST_GPU_HARDWARE").is_none() {
        eprintln!("SKIP half32 native numeric requires VICTORY_TEST_GPU_HARDWARE=1");
        return;
    }
    let (host, _settings) = host_and_settings();
    let profile = super::resolved_shader_profile().unwrap().profile;
    for threads in [[64, 1, 1], [128, 1, 1], [256, 1, 1], [8, 4, 4]] {
        let lowered = lower_native(&native_threads(threads), profile).unwrap();
        assert_eq!(lowered.wave_size, Some(32));
        let shader = HostShader {
            stage: HwStage::Cs,
            profile,
            descriptor_aliases: lowered.descriptor_aliases(),
            direct_heap: lowered.direct_heap,
            dxil: lowered.dxil,
            mesh_fallback: None,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views: Vec::new(),
        };
        let mut validator =
            validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
        validator.validate(&mut shader.dxil.clone()).unwrap();
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let aliases = host.compute_alias_parameters(&shader);
        let count = threads.iter().product::<u32>();
        let data = vec![0; shader.user_data_dwords() as usize];
        let got = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(data))],
                    aliases: &aliases,
                    views: &[View::RawUav(3 * count * 4)],
                    samplers: &[],
                    groups: [3, 1, 1],
                },
            )
            .unwrap();
        let expected = (0..3 * count)
            .flat_map(|global| {
                let flat = global % count;
                [flat, flat & !31, (flat & !31) + 31, global / count]
            })
            .collect::<Vec<_>>();
        assert_eq!(got[0], expected, "threads={threads:?}");
    }
}

#[test]
fn half32_selected_real_fixtures_translate_without_a_device() {
    let Some(paths) = std::env::var_os("VICTORY_HALF32_PROBE") else {
        eprintln!("SKIP no VICTORY_HALF32_PROBE");
        return;
    };
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    struct Restore(u32, Option<u32>);
    impl Drop for Restore {
        fn drop(&mut self) {
            lower::set_host_min_wave(self.0);
            lower::TEST_HOST_WAVE.set(self.1);
        }
    }
    let _restore = Restore(lower::host_min_wave(), lower::TEST_HOST_WAVE.get());
    lower::set_host_min_wave(64);
    lower::TEST_HOST_WAVE.set(Some(64));
    let mut checked = 0;
    for path in std::env::split_paths(&paths) {
        let data = std::fs::read(&path).unwrap();
        let container = crate::Container::parse(&data).unwrap();
        let native = crate::container_programs(&container)
            .unwrap()
            .into_iter()
            .find(|native| native.stage == HwStage::Cs)
            .unwrap();
        assert!(!native.wave64);
        let p = crate::spill::build(&native).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let shader = lower::lower_with_profile(
            &lower::Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: Vec::new(),
                outputs: Vec::new(),
                prims: Vec::new(),
                mesh_stride: 0,
                mesh_fallback: false,
            },
            ShaderProfile::Sm66,
        )
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        eprintln!(
            "CPU translated {}: {} DXIL bytes",
            path.display(),
            shader.dxil.len()
        );
        checked += 1;
    }
    assert!(checked > 0);
}

#[test]
fn half32_extra_wave_operations_are_rejected() {
    for (name, opcode, boolean) in [
        ("waveIsFirstLane", 110, true),
        ("waveActiveOp.i32", 119, false),
    ] {
        let mut s = Shader::new(false, ShaderProfile::Sm66);
        s.half32 = true;
        let ret = if boolean { s.t.i1 } else { s.t.i32 };
        let args = if boolean {
            Vec::new()
        } else {
            vec![s.m.i32(1), s.m.i8(0), s.m.i8(1)]
        };
        s.op(s.entry, name, Attr::NoUnwind, ret, opcode, &args);
        assert!(s.half32_unsupported, "{name}");
    }
}

#[test]
fn half32_bit_reductions_refuse_unknown_kinds_and_mismatched_types() {
    for case in 0..5 {
        let mut s = Shader::new(false, ShaderProfile::Sm66);
        s.half32 = true;
        let b = s.entry;
        let value = if case == 2 { s.m.f32(0) } else { s.m.i32(1) };
        let kind = match case {
            0 => s.m.i8(3),
            1 => s.m.i32(1),
            4 => {
                let one = s.m.i8(1);
                s.m.bin(b, Bin::Xor, one, one)
            }
            _ => s.m.i8(1),
        };
        let ret = if case == 3 { s.t.f32 } else { s.t.i32 };
        s.op(
            b,
            "waveActiveBit.i32",
            Attr::NoUnwind,
            ret,
            120,
            &[value, kind],
        );
        assert!(s.half32_unsupported, "case={case}");
    }
}

#[test]
fn half32_mode_requires_an_exact_physical_wave64_device() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let old_min = lower::host_min_wave();
    let old_max = lower::TEST_HOST_WAVE.get();
    for (minimum, maximum, physical) in [(64, 64, 64), (32, 32, 32), (0, 0, 32), (64, 32, 32)] {
        lower::set_host_min_wave(minimum);
        lower::TEST_HOST_WAVE.set(Some(maximum));
        let lowered = lower_native(&native_threads([64, 1, 1]), ShaderProfile::Sm66).unwrap();
        assert_eq!(lowered.wave_size, Some(32));
        let c = crate::Container::parse(&lowered.dxil).unwrap();
        let psv = c.part(crate::FourCC::PSV0).unwrap().data;
        assert_eq!(
            u32::from_le_bytes(psv[20..24].try_into().unwrap()),
            physical
        );
        assert_eq!(
            u32::from_le_bytes(psv[24..28].try_into().unwrap()),
            physical
        );
    }
    lower::set_host_min_wave(old_min);
    lower::TEST_HOST_WAVE.set(old_max);
}

#[test]
fn half32_lowering_refuses_an_uncovered_wave_reduction() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let old_min = lower::host_min_wave();
    let old_max = lower::TEST_HOST_WAVE.replace(Some(64));
    lower::set_host_min_wave(64);
    let mut native = native_threads([64, 1, 1]);
    let mut words = vec![
        0xBE8C_0305,
        0xBE85_0380,
        0xF408_0202,
        0xFA00_0000,
        0x7E0E_0281,
    ];
    let mut section = super::wave::LATE_OR_SECTION;
    section[0] = 0xBE90_3DC1;
    section[17] = 0xBEFE_0310;
    for word in &mut section {
        if *word & 0xFE00_0000 == 0x3800_0000 {
            *word = (*word & !0x7E00_0000) | (0x25 << 25);
        }
    }
    words.extend(section);
    words.extend([
        0x7E0C_0212,
        0x7E14_0280,
        0xE070_1000,
        0x8002_060A,
        0xBF81_0000,
    ]);
    native.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let result = lower_native(&native, ShaderProfile::Sm66);
    lower::set_host_min_wave(old_min);
    lower::TEST_HOST_WAVE.set(old_max);
    assert!(matches!(
        result,
        Err(ShaderError::Unsupported {
            what: "a wave operation outside the logical wave32 fallback",
            ..
        })
    ));
}

#[test]
fn half32_whole_ballots_are_rejected_before_serialization() {
    for consumer in ["select", "call", "phi", "metadata"] {
        let mut s = Shader::new(false, ShaderProfile::Sm66);
        s.half32 = true;
        s.threads = Some([256, 1, 1]);
        s.wave_size = Some(32);
        let four = s.m.named_struct("dx.types.fouri32", vec![s.t.i32; 4]);
        let yes = s.m.bool(true);
        let ballot = s.op(
            s.entry,
            "waveActiveBallot",
            Attr::NoUnwind,
            four,
            116,
            &[yes],
        );
        match consumer {
            "select" => {
                s.m.select(s.entry, yes, ballot, ballot);
            }
            "call" => {
                let ty = s.m.func_ty(s.t.void, vec![four]);
                s.m.ptr(ty);
                let function = s.m.function("aggregate_consumer", ty, None, false);
                s.m.call(s.entry, function, vec![ballot]);
            }
            "phi" => {
                let after = s.m.new_block();
                let phi = s.m.phi(after, four);
                s.m.set_phi(phi, vec![(ballot, s.entry)]);
                s.m.terminate(s.entry, Term::Br(after));
            }
            "metadata" => {
                let value = s.m.md_value(ballot);
                s.m.named_md("half32.escape", vec![value]);
            }
            _ => unreachable!(),
        }
        assert_eq!(
            s.try_finish(),
            Err("a logical wave32 ballot used as an aggregate"),
            "{consumer}"
        );
    }
}

fn native_permlanex16(modifiers: u32) -> NativeProgram {
    let mut native = native_threads([128, 1, 1]);
    native.code = [
        0xBE8C_0305,
        0xBE85_0380,
        0xF408_0202,
        0xFA00_0000,
        0x7E06_0300,
        0x7E0C_02FF,
        0x1000,
        vop2(0x25, 6, 0x106, 3),
        0x7E0E_02FF,
        0x8000,
        vop2(0x25, 7, 0x107, 3),
        0xBEFE_0300,
        0xD778_0007 | modifiers,
        0x0008_0306,
        0xBEFE_03C1,
        0x7E0A_020C,
        vop2(0x1a, 5, 135, 5),
        vop2(0x25, 5, 0x105, 3),
        vop2(0x1a, 10, 130, 5),
        0xE070_1000,
        0x8002_070A,
        0xBF81_0000,
    ]
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect();
    native
}

#[test]
fn half32_permlanex16_uses_guarded_sources_or_the_old_destination() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if std::env::var_os("VICTORY_TEST_GPU_HARDWARE").is_none() {
        eprintln!("SKIP permlanex16 numeric requires VICTORY_TEST_GPU_HARDWARE=1");
        return;
    }
    let (host, _settings) = host_and_settings();
    let profile = super::resolved_shader_profile().unwrap().profile;
    let lowered = lower_native(&native_permlanex16(0), profile).unwrap();
    let shader = HostShader {
        stage: HwStage::Cs,
        profile,
        descriptor_aliases: lowered.descriptor_aliases(),
        direct_heap: lowered.direct_heap,
        dxil: lowered.dxil,
        mesh_fallback: None,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    let mut validator =
        validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut shader.dxil.clone()).unwrap();
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let aliases = host.compute_alias_parameters(&shader);
    for mask in [0, u32::MAX, 1, 1 << 16, 0xAAAA_5555, 0x8001_8001] {
        for (lo, hi) in [
            (0, 0),
            (0x7654_3210, 0xFEDC_BA98),
            (0x89AB_CDEF, 0x0123_4567),
            (0x2031_7465, 0xA8B9_CEDF),
        ] {
            let got = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(vec![mask, lo, hi, 0]))],
                        aliases: &aliases,
                        views: &[View::RawUav(3 * 128)],
                        samplers: &[],
                        groups: [3, 1, 1],
                    },
                )
                .unwrap();
            let selector = u64::from(lo) | u64::from(hi) << 32;
            let expected = (0..3 * 128)
                .map(|global| {
                    let flat = global % 128;
                    let lane = flat & 31;
                    let source =
                        ((lane & !15) ^ 16) | ((selector >> (4 * (lane & 15)) & 15) as u32);
                    if mask >> lane & 1 != 0 && mask >> source & 1 != 0 {
                        (flat & !31) + source + 0x1000
                    } else {
                        flat + 0x8000
                    }
                })
                .collect::<Vec<u32>>();
            assert_eq!(
                got[0], expected,
                "mask={mask:08x}, selectors={lo:08x}/{hi:08x}"
            );
        }
    }
}

#[test]
fn half32_permlanex16_rejects_unimplemented_modifiers() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for modifiers in [1 << 11, 1 << 12, 1 << 15, 1 << 8] {
        assert!(matches!(
            lower_native(&native_permlanex16(modifiers), ShaderProfile::Sm66),
            Err(ShaderError::Unsupported {
                what: "permlane input modifiers",
                ..
            })
        ));
    }
}

pub(super) fn native_packed_fmac() -> NativeProgram {
    let mut native = native_threads([128, 1, 1]);
    native.code = [
        0xBE8C_0305,
        0xBE85_0380,
        0xF408_0202,
        0xFA00_0000,
        0x7E06_0300,
        0x7E0C_0200,
        0x7E0E_0201,
        0x7E10_0202,
        0xBEFE_0303,
        vop2(0x3c, 8, 0x106, 7),
        0xBEFE_03C1,
        0x7E0A_020C,
        vop2(0x1a, 5, 135, 5),
        vop2(0x25, 5, 0x105, 3),
        vop2(0x1a, 10, 130, 5),
        0xE070_1000,
        0x8002_080A,
        0xBF81_0000,
    ]
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect();
    native
}

pub(super) fn shader_from_lowered(lowered: lower::Lowered, profile: ShaderProfile) -> HostShader {
    HostShader {
        stage: HwStage::Cs,
        profile,
        descriptor_aliases: lowered.descriptor_aliases(),
        direct_heap: lowered.direct_heap,
        dxil: lowered.dxil,
        mesh_fallback: None,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    }
}

#[test]
fn packed_fmac_reads_both_halves_of_the_old_destination_and_exec() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if std::env::var_os("VICTORY_TEST_GPU_HARDWARE").is_none() {
        eprintln!("SKIP packed fmac numeric requires VICTORY_TEST_GPU_HARDWARE=1");
        return;
    }
    let (host, _settings) = host_and_settings();
    let profile = super::resolved_shader_profile().unwrap().profile;
    let shader = shader_from_lowered(
        lower_native(&native_packed_fmac(), profile).unwrap(),
        profile,
    );
    let mut validator =
        validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut shader.dxil.clone()).unwrap();
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let aliases = host.compute_alias_parameters(&shader);
    for (a, b, old, result) in [
        (0x4000_3C00, 0x4400_4200, 0x4600_4500, 0x4B00_4800),
        (0x4200_C000, 0x4400_3800, 0xBC00_4000, 0x4980_3C00),
        (0x7BFF_7BFF, 0x4000_4000, 0, 0x7C00_7C00),
        (0x3C00_3C00, 0x1000_1000, 0x3C01_3C00, 0x3C02_3C00),
        (0x3C00_3C00, 0x4000_4000, 0x4400_4200, 0x4600_4500),
        (0xFBFF_FBFF, 0x4000_4000, 0, 0xFC00_FC00),
    ] {
        for mask in [0, u32::MAX, 1, 1 << 31, 0xAAAA_5555] {
            let got = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(vec![a, b, old, mask]))],
                        aliases: &aliases,
                        views: &[View::RawUav(3 * 128)],
                        samplers: &[],
                        groups: [3, 1, 1],
                    },
                )
                .unwrap();
            let expected = (0..3 * 128)
                .map(|flat| {
                    if mask >> (flat & 31) & 1 != 0 {
                        result
                    } else {
                        old
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(
                got[0], expected,
                "a={a:08x}, b={b:08x}, old={old:08x}, mask={mask:08x}"
            );
        }
    }
}

pub(super) fn native_a16_atomic(d16: bool) -> NativeProgram {
    let mut native = native_threads([64, 1, 1]);
    native.used[0].dwords = 8;
    native.used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 8,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    });
    native.code = [
        0xBE8C_0305,
        0xBE85_0380,
        0xF40C_0202,
        0xFA00_0000,
        0xF408_0402,
        0xFA00_0020,
        0x7E06_0300,
        vop2(0x1b, 4, 135, 3),
        vop2(0x16, 5, 131, 3),
        vop2(0x1a, 5, 144, 5),
        vop2(0x1c, 12, 0x104, 5),
        0x7E0A_02FF,
        123,
        0xF03C_3108,
        0x4002_050C | if d16 { 1 << 31 } else { 0 },
        vop2(0x25, 5, 129, 3),
        0xF044_3108,
        0x4002_050C | if d16 { 1 << 31 } else { 0 },
        vop2(0x1a, 10, 130, 3),
        0xE070_1000,
        0x8004_050A,
        0xBF81_0000,
    ]
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect();
    native
}

#[test]
fn a16_image_atomic_unpacks_xy_and_keeps_32_bit_data_and_return_value() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if std::env::var_os("VICTORY_TEST_GPU_HARDWARE").is_none() {
        eprintln!("SKIP A16 atomic numeric requires VICTORY_TEST_GPU_HARDWARE=1");
        return;
    }
    let (host, _settings) = host_and_settings();
    let profile = super::resolved_shader_profile().unwrap().profile;
    let shader = shader_from_lowered(
        lower_native(&native_a16_atomic(false), profile).unwrap(),
        profile,
    );
    let mut validator =
        validator::Validator::load(validator::Validator::default_path().as_deref()).unwrap();
    validator.validate(&mut shader.dxil.clone()).unwrap();
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let aliases = host.compute_alias_parameters(&shader);
    let got = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(vec![0; 4]))],
                aliases: &aliases,
                views: &[View::UintTextureUav(8, 8), View::RawUav(64)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap();
    assert_eq!(got[0], (0..64).map(|flat| 124 + flat).collect::<Vec<_>>());
    assert_eq!(got[1], vec![123; 64]);
}

#[test]
fn d16_image_atomics_remain_rejected() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    assert!(matches!(
        lower_native(&native_a16_atomic(true), ShaderProfile::Sm66),
        Err(ShaderError::Unsupported {
            what: "a 16-bit image atomic",
            ..
        })
    ));
}
