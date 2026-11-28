use super::half32_tests::{
    host_and_settings, lower_native, native_packed_fmac, shader_from_lowered,
};
use super::host::{Dispatch, Root, View};
use super::validator;

fn half_is_nan(bits: u16) -> bool {
    bits & 0x7C00 == 0x7C00 && bits & 0x03FF != 0
}

fn half_units(bits: u16) -> i128 {
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    let magnitude = if exponent == 0 {
        i128::from(fraction)
    } else {
        i128::from(1024 + fraction) << (exponent - 1)
    };
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn exact_fmac(a: u16, b: u16, accumulator: u16) -> u16 {
    if [a, b, accumulator].into_iter().any(half_is_nan) {
        return 0x7E00;
    }
    let infinite = |x: u16| x & 0x7FFF == 0x7C00;
    let zero = |x: u16| x & 0x7FFF == 0;
    let product_sign = (a ^ b) & 0x8000;
    if infinite(a) || infinite(b) {
        if zero(a) || zero(b) || infinite(accumulator) && (accumulator & 0x8000) != product_sign {
            return 0x7E00;
        }
        return product_sign | 0x7C00;
    }
    if infinite(accumulator) {
        return accumulator;
    }
    let product = half_units(a) * half_units(b);
    let addend = half_units(accumulator) << 24;
    let exact = product + addend;
    if exact == 0 {
        return if product == 0 && addend == 0 && product_sign & accumulator & 0x8000 != 0 {
            0x8000
        } else {
            0
        };
    }
    let sign = if exact < 0 { 0x8000 } else { 0 };
    let exact = exact.unsigned_abs();
    if exact >= 65520u128 << 48 {
        return sign | 0x7C00;
    }
    let (mut lo, mut hi) = (0u16, 0x7BFFu16);
    while lo < hi {
        let middle = lo + (hi - lo).div_ceil(2);
        if (half_units(middle) as u128) << 24 <= exact {
            lo = middle;
        } else {
            hi = middle - 1;
        }
    }
    let below = (half_units(lo) as u128) << 24;
    let next = lo + 1;
    let above = if next == 0x7C00 {
        1u128 << 64
    } else {
        (half_units(next) as u128) << 24
    };
    let under = exact - below;
    let over = above - exact;
    sign | if under > over || under == over && lo & 1 != 0 {
        next
    } else {
        lo
    }
}

fn packed_exact(a: u32, b: u32, accumulator: u32) -> u32 {
    u32::from(exact_fmac(a as u16, b as u16, accumulator as u16))
        | (u32::from(exact_fmac(
            (a >> 16) as u16,
            (b >> 16) as u16,
            (accumulator >> 16) as u16,
        )) << 16)
}

#[test]
fn packed_fmac_integer_oracle_covers_half_boundaries_and_fused_ties() {
    for (a, b, accumulator, expected) in [
        (0x3C00, 0x4200, 0x4500, 0x4800),
        (0x4000, 0x4400, 0x4600, 0x4B00),
        (0x3E00, 0x6956, 0x0400, 0x6C01),
        (0x3E00, 0x6956, 0x8400, 0x6C00),
        (0xBE00, 0x6956, 0x8400, 0xEC01),
        (0xBE00, 0x6956, 0x0400, 0xEC00),
        (0x7BFF, 0x4000, 0x0000, 0x7C00),
        (0xFBFF, 0x4000, 0x0000, 0xFC00),
        (0x7BFF, 0x3C00, 0x4C00, 0x7C00),
        (0x7BFF, 0x3C00, 0x4BFF, 0x7BFF),
        (0x0001, 0x3C00, 0x0000, 0x0001),
        (0x0001, 0x3800, 0x0000, 0x0000),
        (0x0003, 0x3800, 0x0000, 0x0002),
        (0x8001, 0x3800, 0x0000, 0x8000),
        (0x8003, 0x3800, 0x0000, 0x8002),
        (0x8000, 0x3C00, 0x8000, 0x8000),
        (0x8000, 0x3C00, 0x0000, 0x0000),
        (0x3C00, 0x3C00, 0xBC00, 0x0000),
        (0x7C00, 0x3C00, 0x3C00, 0x7C00),
        (0xFC00, 0x3C00, 0x3C00, 0xFC00),
    ] {
        assert_eq!(exact_fmac(a, b, accumulator), expected);
    }
    for (a, b, accumulator) in [
        (0x7E00, 0x3C00, 0),
        (0x7C00, 0, 0),
        (0x7C00, 0x3C00, 0xFC00),
    ] {
        assert!(half_is_nan(exact_fmac(a, b, accumulator)));
    }
}

#[test]
fn packed_fmac_matches_exact_half_rne_with_masks_and_independent_halves() {
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
    let mut cases = vec![
        (0x4000_3C00, 0x4400_4200, 0x4600_4500),
        (0xBE00_3E00, 0x6956_6956, 0x8400_0400),
        (0xBE00_3E00, 0x6956_6956, 0x0400_8400),
        (0x7BFF_7BFF, 0x4000_4000, 0),
        (0xFBFF_FBFF, 0x4000_4000, 0),
        (0x7BFF_7BFF, 0x3C00_3C00, 0x4BFF_4C00),
        (0x0003_0001, 0x3800_3800, 0),
        (0x8003_8001, 0x3800_3800, 0),
        (0x03FF_0400, 0x3C00_3C00, 0x0001_8001),
        (0x0001_0001, 0x0001_0001, 0),
        (0x8000_8000, 0x3C00_3C00, 0x0000_8000),
        (0x3C00_3C00, 0x3C00_3C00, 0xBC00_BC00),
        (0x7C00_FC00, 0x3C00_3C00, 0x3C00_3C00),
        (0x7C00_7E00, 0x0000_3C00, 0),
        (0x3C00_3C00, 0x3C00_3C00, 0xFE01_7E01),
    ];
    let mut state = 0xF16F_4ACCu32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let mut half = state as u16;
        if half & 0x7C00 == 0x7C00 {
            half &= 0xFBFF;
        }
        u32::from(half)
    };
    for _ in 0..128 {
        let mut packed = || next() | (next() << 16);
        cases.push((packed(), packed(), packed()));
    }
    let mut checked = 0usize;
    for (a, b, old) in cases {
        let expected = packed_exact(a, b, old);
        for mask in [0u32, u32::MAX, 1, 1 << 31, 0xAAAA_5555] {
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
            assert_eq!(got[0].len(), 3 * 128);
            for (flat, &actual) in got[0].iter().enumerate() {
                if mask >> (flat & 31) & 1 == 0 {
                    assert_eq!(actual, old, "inactive lane={flat}, mask={mask:08x}");
                    checked += 2;
                    continue;
                }
                let wanted = expected;
                for shift in [0, 16] {
                    let (actual, wanted) = ((actual >> shift) as u16, (wanted >> shift) as u16);
                    if half_is_nan(wanted) {
                        assert!(half_is_nan(actual));
                    } else {
                        assert_eq!(
                            actual, wanted,
                            "a={a:08x} b={b:08x} old={old:08x} mask={mask:08x} lane={flat} half={shift}"
                        );
                    }
                    checked += 1;
                }
            }
        }
    }
    eprintln!("packed fmac independent exact half comparisons: {checked}");
}
