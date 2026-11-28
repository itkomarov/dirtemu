use super::half32_tests::{
    host_and_settings, lower_native, native_a16_atomic, shader_from_lowered,
};
use super::host::{Dispatch, Root, View};
use super::{ShaderProfile, validator};
use crate::ShaderError;

fn counter_program(d16: bool, value: u32, atomic: bool) -> crate::native::NativeProgram {
    let mut native = native_a16_atomic(false);
    let mut words = native
        .code
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    let at = words.iter().position(|&word| word == 0xF03C_3108).unwrap();
    assert_eq!(words[at - 2], 0x7E0A_02FF);
    words[at - 1] = value;
    words[at] = 0xF020_3108;
    if d16 {
        words[at + 1] |= 1 << 31;
    }
    if !atomic {
        let at = words.iter().position(|&word| word == 0xF044_3108).unwrap();
        words.drain(at..at + 2);
    }
    native.code = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    native
}

#[test]
fn counter_store_keeps_integer_bits_and_packed_coordinates() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if std::env::var_os("VICTORY_TEST_GPU_HARDWARE").is_none() {
        eprintln!("SKIP counter store requires VICTORY_TEST_GPU_HARDWARE=1");
        return;
    }
    let (host, _settings) = host_and_settings();
    let profile = super::resolved_shader_profile().unwrap().profile;
    for (d16, value) in [(false, 123), (false, 0x3F80_0001), (true, 123)] {
        let native = counter_program(d16, value, true);
        let shader = shader_from_lowered(lower_native(&native, profile).unwrap(), profile);
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
        assert_eq!(
            got[0],
            (0..64).map(|flat| value + flat + 1).collect::<Vec<_>>(),
            "d16={d16}"
        );
        assert_eq!(got[1], vec![value; 64], "d16={d16}");
    }
}

#[test]
fn single_channel_store_needs_a_proven_32_bit_counter() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    assert!(matches!(
        lower_native(&counter_program(false, 123, false), ShaderProfile::Sm66),
        Err(ShaderError::Unsupported {
            what: "an image store of some channels",
            ..
        })
    ));
    assert!(matches!(
        lower_native(&counter_program(true, 0x8000, true), ShaderProfile::Sm66),
        Err(ShaderError::Unsupported {
            what: "a 16-bit image store of one channel",
            ..
        })
    ));
}
