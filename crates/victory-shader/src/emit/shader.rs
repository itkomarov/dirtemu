//! A DXIL vertex, pixel, compute or mesh shader under construction: the module,
//! the `dx.op` declarations it calls, its signatures and bound resources,
//! and the metadata and container that wrap it up (SM 6.6, validator 1.6).

use super::bitcode::{Attr, Md, Module, Ty, V};
use super::container::{self, Binding, Psv, SigElement};

/// Where the shader reads its user data: `b0` in this space.
pub const USER_DATA_SPACE: u32 = 0;
/// Where immediate samplers are bound: `s<n>` in this space.
pub const SAMPLER_SPACE: u32 = 0;
/// Where root CBVs and SRVs are bound: `b<root parameter index>` and
/// `t<root parameter index>` in this space.
pub const ROOT_CBV_SPACE: u32 = 1;

/// The root SRV `t<SLOT_STRIDES>` in [`ROOT_CBV_SPACE`] of a shader that
/// reads buffers through descriptors it picks from the view heap at an
/// index it computes (SM 6.6 dynamic resources): a raw buffer of
/// [`SLOT_STRIDES_COUNT`] dwords the host keeps, dword `i` the bytes per
/// element of the buffer view at heap slot `i` (0: none or not a buffer).
/// The console's descriptor holds that stride and its hardware multiplies
/// the element index by it; the program's code does not say it, and the
/// used-resource list only names what tables hold. A shader that has this
/// root SRV also tells the host that the buffer views of the whole heap
/// must be ready when it runs, not only those its tables name.
pub const SLOT_STRIDES: u8 = 255;
/// Dwords of the [`SLOT_STRIDES`] buffer: heap indices are clamped to it.
pub const SLOT_STRIDES_COUNT: u32 = 1 << 21;

/// Title memory by address (`resource::MEMORY`): console code reads and
/// writes through raw 40-bit addresses it takes from root constants,
/// descriptors and other memory (the ray-tracing structure builder and the
/// programs that walk its structures). The host keeps copies of the title
/// memory shaders use; a shader with such an access finds the copy by the
/// address:
///
/// - the root SRV `t<SLOT_MEMORY>` in [`ROOT_CBV_SPACE`] is a directory of
///   [`MEMORY_PAGES`] entries of two dwords, one per 64 KiB of title
///   address space: the number of the host copy holding that page (0: none
///   yet) and the page's byte offset in that copy;
/// - copy `n` is the raw buffer view at view-heap slot `base + n`, `base`
///   the user-data dword of kind [`USER_MEMORY_BASE`]; slot `base` itself
///   reads zeros and drops writes;
/// - the root UAV `u<SLOT_FAULTS>` takes the page numbers of accesses that
///   found no copy ([`MEMORY_FAULTS`] dwords, dword 0 a sink), which the
///   host reads afterwards to make those copies for the frames to come.
pub const SLOT_MEMORY: u8 = 254;
/// See [`SLOT_MEMORY`].
pub const SLOT_FAULTS: u8 = 253;
/// The compute-emulated mesh groups' output records (`u` for the compute
/// pass, `t` for the vertex-pulling pass), in [`ROOT_CBV_SPACE`].
pub const SLOT_MESH_OUTPUT: u8 = 252;
/// `RootType::Other` code of the first mesh group of a direct fallback
/// dispatch. Indirect dispatches start at zero.
pub const USER_MESH_GROUP_BASE: u32 = 0x104;
/// Entries of the [`SLOT_MEMORY`] directory: title addresses below 128 GiB.
pub const MEMORY_PAGES: u32 = 1 << 21;
/// Dwords of the [`SLOT_FAULTS`] buffer.
pub const MEMORY_FAULTS: u32 = 4096;
/// `RootType::Other` code of a user-data dword the host computes: the view
/// heap slot of the memory copies' views ([`SLOT_MEMORY`]).
pub const USER_MEMORY_BASE: u32 = 0x100;
/// `RootType::Other` code of a user-data dword the host computes: the low
/// 32 bits of the title address of the buffer view at dword `dword` of the
/// descriptor table of root parameter `slot`; [`USER_VIEW_ADDRESS`] + 1:
/// its bits from 32 up. (Console code reads the address out of a buffer
/// descriptor and uses it as a pointer.)
pub const USER_VIEW_ADDRESS: u32 = 0x101;
/// `RootType::Other` code of a user-data dword the host computes: the
/// bytes (0 to 15) by which the raw view the host binds for the root view
/// of root parameter `slot` begins before the title's address. A raw view
/// begins on 16 bytes of its host buffer; the console's root views begin
/// anywhere (the structure builder's counters sit at addresses ending in
/// 4 and c). The shader adds the dword to every byte offset through that
/// view. Asked for every root UAV, and for the root SRVs of programs with
/// a spilled user-data table.
pub const USER_VIEW_BIAS: u32 = 0x103;

pub const DYNAMIC_SPACE_BASE: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum DescriptorClass {
    Srv = 0,
    Uav = 1,
    Cbv = 2,
    Sampler = 3,
}

impl DescriptorClass {
    pub fn code(self) -> u8 {
        self as u8
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingKey {
    pub class: DescriptorClass,
    pub kind: u32,
    pub stride: u32,
    pub component: u32,
    pub counter: bool,
}

impl BindingKey {
    fn from_props(sampler: bool, props: (u32, u32)) -> Self {
        let class = if sampler {
            DescriptorClass::Sampler
        } else if props.0 & 0xf == 13 {
            DescriptorClass::Cbv
        } else if props.0 & 0x1000 != 0 {
            DescriptorClass::Uav
        } else {
            DescriptorClass::Srv
        };
        let kind = if sampler {
            props.0 & 0x8fff
        } else {
            props.0 & 0xfff
        };
        Self {
            class,
            kind,
            counter: class == DescriptorClass::Uav && props.0 & 0x8000 != 0,
            stride: if class == DescriptorClass::Cbv || kind == 12 {
                props.1
            } else {
                0
            },
            component: if matches!(class, DescriptorClass::Cbv | DescriptorClass::Sampler)
                || kind == 12
            {
                0
            } else {
                props.1
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorTable {
    pub key: BindingKey,
    pub space: u32,
    pub range_id: u32,
    pub upper: u32,
}

/// A descriptor-table alias of the dynamic heap for SM 6.5, in a space of
/// its own: the host binds each alias to the base of the title's heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorAlias {
    pub class: DescriptorClass,
    pub space: u32,
}

/// One `dx.resources` record while the lists are assembled.
struct Made {
    range: u32,
    node: Md,
    binding: Binding,
}

/// The `PSVResourceType` of a resource range (1 sampler, 2 CBV, 3 typed
/// SRV, 4 raw SRV, 5 structured SRV, 6 typed UAV, 7 raw UAV, 8 structured
/// UAV, 9 structured UAV with counter).
fn psv_resource_type(key: BindingKey) -> u32 {
    match key.class {
        DescriptorClass::Sampler => 1,
        DescriptorClass::Cbv => 2,
        DescriptorClass::Srv => match key.kind {
            11 => 4,
            12 => 5,
            _ => 3,
        },
        DescriptorClass::Uav => match key.kind {
            11 => 7,
            12 if key.counter => 9,
            12 => 8,
            _ => 6,
        },
    }
}

fn physical_wave_size(
    logical: Option<u32>,
    threads: Option<[u32; 3]>,
    minimum: u32,
) -> Option<u32> {
    if logical == Some(32)
        && minimum == 64
        && threads.is_some_and(|size| (1..=32).contains(&size.iter().product::<u32>()))
    {
        Some(64)
    } else {
        logical
    }
}

#[cfg(test)]
mod physical_wave_tests {
    use super::physical_wave_size;

    #[test]
    fn a_single_small_logical_wave_can_occupy_a_wider_host_wave() {
        for threads in [[1, 1, 1], [8, 1, 1], [16, 1, 1], [32, 1, 1], [8, 4, 1]] {
            assert_eq!(physical_wave_size(Some(32), Some(threads), 64), Some(64));
            assert_eq!(physical_wave_size(Some(32), Some(threads), 32), Some(32));
        }
    }

    #[test]
    fn multiple_logical_waves_are_never_combined_by_metadata() {
        for threads in [[64, 1, 1], [160, 1, 1], [16, 16, 1]] {
            assert_eq!(physical_wave_size(Some(32), Some(threads), 64), Some(32));
        }
        assert_eq!(physical_wave_size(Some(32), None, 64), Some(32));
        assert_eq!(physical_wave_size(Some(64), Some([64, 1, 1]), 64), Some(64));
        assert_eq!(physical_wave_size(None, Some([32, 1, 1]), 64), None);
    }
}

/// `ShaderFlags` bits (metadata) and their `SFI0` feature bits.
pub(crate) mod flags {
    pub const RAW_BUFFERS: u64 = 1 << 4;
    pub const STENCIL_REF: u64 = 1 << 11;
    pub const TYPED_UAV_LOADS: u64 = 1 << 13;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const BARYCENTRICS: u64 = 1 << 22;
    pub const INT64: u64 = 1 << 20;
    pub const ATOMIC_TYPED64: u64 = 1 << 27;
    pub const ATOMIC_HEAP64: u64 = 1 << 32;
    pub const RESOURCE_HEAP: u64 = 1 << 30;
    pub const SAMPLER_HEAP: u64 = 1 << 31;
}

pub(crate) struct Types {
    pub void: Ty,
    pub i1: Ty,
    pub i32: Ty,
    pub f32: Ty,
    pub handle: Ty,
    pub props: Ty,
    pub bind: Ty,
    pub cbuf_ret: Ty,
    pub res_ret: Ty,
}

/// What a mesh shader declares beyond its thread-group size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mesh {
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
}

/// What a geometry shader declares: triangles in, triangle strips out on
/// stream 0, one instance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Geometry {
    pub max_vertices: u32,
}

/// The null slot a host view heap keeps at its end when the title's heap did not fit in it
/// (0: no limit). Shaders translated while it is set read that slot for any view index at or
/// past it: a host that caps its heap (NVIDIA at 1 000 000 views; a retail game's heap is
/// larger) hung its GPU on reads past the end (GPU-based validation: "descriptor heap index
/// out of bounds"), which AMD's driver forgives.
static VIEW_LIMIT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// See [`VIEW_LIMIT`]: `null_slot` is the last slot of the host's view heap, kept null.
pub fn set_view_heap_limit(null_slot: u32) {
    VIEW_LIMIT.store(null_slot, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn view_heap_limit() -> u32 {
    VIEW_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) struct Shader {
    pub m: Module,
    pub t: Types,
    pub profile: super::ShaderProfile,
    pub descriptor_tables: Vec<DescriptorTable>,
    next_ranges: [u32; 4],
    user_data_range: Option<u32>,
    root_cbv_ranges: Vec<u32>,
    root_srv_ranges: Vec<u32>,
    root_uav_ranges: Vec<u32>,
    sampler_ranges: Vec<u32>,
    #[cfg(test)]
    pub opcodes: Vec<u32>,
    /// Entry-point property tags (tag 0 flags, 1 geometry, 4/9 groups,
    /// 11 wave size), in emission order.
    pub property_tags: Vec<u32>,
    pub pixel: bool,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// A mesh shader's per-primitive outputs.
    pub prims: Vec<SigElement>,
    /// A mesh shader (`threads` is its thread-group size).
    pub mesh: Option<Mesh>,
    /// A geometry shader (the rectangle stage, `rect.rs`); `pixel` is false.
    pub geometry: Option<Geometry>,
    /// Dwords of user data the shader reads (0: no user-data buffer).
    pub user_data: u32,
    /// Immediate samplers, bound as `s0..` in [`SAMPLER_SPACE`].
    pub samplers: u32,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    /// Root CBVs read, by root parameter index, with their handles.
    pub root_cbvs: Vec<(u8, V)>,
    /// Root SRVs (raw buffers) read, likewise.
    pub root_srvs: Vec<(u8, V)>,
    /// Root UAVs (raw buffers) read or written, likewise.
    pub root_uavs: Vec<(u8, V)>,
    pub flags: u64,
    /// Heap handles made with a non-uniform index.
    pub nonuniform_handles: u32,
    /// A compute shader with this thread-group size.
    pub threads: Option<[u32; 3]>,
    /// The wave size a compute shader requires (`WaveSize`).
    pub wave_size: Option<u32>,
    pub half32: bool,
    pub half32_unsupported: bool,
    /// Thread-group shared memory, in dwords.
    pub lds_dwords: u32,
    lds: Option<V>,
    main: V,
    pub entry: usize,
}

impl Shader {
    pub fn new(pixel: bool, profile: super::ShaderProfile) -> Self {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i8 = m.int(8);
        let i32 = m.int(32);
        let f32 = m.float();
        let i8p = m.ptr(i8);
        let handle = m.named_struct("dx.types.Handle", vec![i8p]);
        let props = m.named_struct("dx.types.ResourceProperties", vec![i32, i32]);
        let bind = m.named_struct("dx.types.ResBind", vec![i32, i32, i32, i8]);
        let cbuf_ret = m.named_struct("dx.types.CBufRet.i32", vec![i32, i32, i32, i32]);
        let res_ret = m.named_struct("dx.types.ResRet.f32", vec![f32, f32, f32, f32, i32]);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        let main = m.function("main", main_ty, None, true);
        let entry = m.new_block();
        Shader {
            m,
            profile,
            descriptor_tables: Vec::new(),
            next_ranges: [0; 4],
            user_data_range: None,
            root_cbv_ranges: Vec::new(),
            root_srv_ranges: Vec::new(),
            root_uav_ranges: Vec::new(),
            sampler_ranges: Vec::new(),
            #[cfg(test)]
            opcodes: Vec::new(),
            property_tags: Vec::new(),
            t: Types {
                void,
                i1,
                i32,
                f32,
                handle,
                props,
                bind,
                cbuf_ret,
                res_ret,
            },
            pixel,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh: None,
            geometry: None,
            user_data: 0,
            samplers: 0,
            cmp_samplers: 0,
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            flags: 0,
            nonuniform_handles: 0,
            threads: None,
            wave_size: None,
            half32: false,
            half32_unsupported: false,
            lds_dwords: 0,
            lds: None,
            main,
            entry,
        }
    }

    /// The thread-group shared memory: `[lds_dwords x i32]` in address
    /// space 3.
    pub fn lds(&mut self) -> V {
        if let Some(g) = self.lds {
            return g;
        }
        let arr = self.m.array(self.lds_dwords, self.t.i32);
        let g = self.m.global("lds", arr, 3);
        self.lds = Some(g);
        g
    }

    /// Calls `dx.op.<name>` (class and overload) with `opcode` and `args`.
    pub fn op(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        if self.half32 && (name.starts_with("wave") || name.starts_with("quad")) {
            let valid = match (opcode, name) {
                (111, "waveGetLaneIndex") => ret == self.t.i32 && args.is_empty(),
                (116, "waveActiveBallot") => {
                    let four = self.m.named_struct("dx.types.fouri32", vec![self.t.i32; 4]);
                    ret == four && args.len() == 1 && self.m.type_of(args[0]) == self.t.i1
                }
                (117, "waveReadLaneAt.i32") => {
                    ret == self.t.i32
                        && args.len() == 2
                        && args.iter().all(|arg| self.m.type_of(*arg) == self.t.i32)
                }
                (118, "waveReadLaneFirst.i32") => {
                    ret == self.t.i32 && args.len() == 1 && self.m.type_of(args[0]) == self.t.i32
                }
                (120, "waveActiveBit.i32") => {
                    let i8t = self.m.int(8);
                    ret == self.t.i32
                        && args.len() == 2
                        && self.m.type_of(args[0]) == self.t.i32
                        && self.m.type_of(args[1]) == i8t
                        && matches!(self.m.int_value(args[1]), Some(0..=2))
                }
                _ => false,
            };
            if valid {
                return self.half32_op(block, ret, opcode, args);
            }
            #[cfg(test)]
            eprintln!("logical wave32 uncovered operation: {name} opcode={opcode}");
            self.half32_unsupported = true;
        }
        self.raw_op(block, name, attr, ret, opcode, args)
    }

    fn physical_lane(&mut self, block: usize) -> V {
        self.raw_op(
            block,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.t.i32,
            111,
            &[],
        )
    }

    fn half32_target(&mut self, block: usize, lane: V) -> V {
        let physical = self.physical_lane(block);
        let mask = self.m.i32(!31);
        let base = self.m.bin(block, super::bitcode::Bin::And, physical, mask);
        self.m.bin(block, super::bitcode::Bin::Add, base, lane)
    }

    fn half32_ballot(&mut self, block: usize, condition: V) -> V {
        let four = self.m.named_struct("dx.types.fouri32", vec![self.t.i32; 4]);
        let ballot = self.raw_op(
            block,
            "waveActiveBallot",
            Attr::NoUnwind,
            four,
            116,
            &[condition],
        );
        let low = self.m.extract(block, ballot, 0);
        let high = self.m.extract(block, ballot, 1);
        let lane = self.physical_lane(block);
        let boundary = self.m.i32(32);
        let upper = self
            .m
            .cmp(block, super::bitcode::pred::IUGE, lane, boundary);
        self.m.select(block, upper, high, low)
    }

    fn half32_op(&mut self, block: usize, ret: Ty, opcode: u32, args: &[V]) -> V {
        match opcode {
            111 => {
                let lane = self.physical_lane(block);
                let mask = self.m.i32(31);
                self.m.bin(block, super::bitcode::Bin::And, lane, mask)
            }
            116 => {
                let mask = self.half32_ballot(block, args[0]);
                let zero = self.m.i32(0);
                let mut result = self.m.undef(ret);
                for (index, value) in [mask, zero, zero, zero].into_iter().enumerate() {
                    result = self.m.insert(block, result, value, index as u32);
                }
                result
            }
            117 => {
                let target = self.half32_target(block, args[1]);
                self.raw_op(
                    block,
                    "waveReadLaneAt.i32",
                    Attr::NoUnwind,
                    ret,
                    117,
                    &[args[0], target],
                )
            }
            118 => {
                let yes = self.m.bool(true);
                let mask = self.half32_ballot(block, yes);
                let first = self.raw_op(
                    block,
                    "unaryBits.i32",
                    Attr::ReadNone,
                    self.t.i32,
                    32,
                    &[mask],
                );
                let target = self.half32_target(block, first);
                self.raw_op(
                    block,
                    "waveReadLaneAt.i32",
                    Attr::NoUnwind,
                    ret,
                    117,
                    &[args[0], target],
                )
            }
            120 => {
                let identity = match self.m.int_value(args[1]) {
                    Some(0) => u32::MAX,
                    Some(1 | 2) => 0,
                    _ => unreachable!(),
                };
                let identity = self.m.i32(identity);
                let lane = self.physical_lane(block);
                let boundary = self.m.i32(32);
                let upper = self
                    .m
                    .cmp(block, super::bitcode::pred::IUGE, lane, boundary);
                let low_arg = self.m.select(block, upper, identity, args[0]);
                let high_arg = self.m.select(block, upper, args[0], identity);
                let low = self.raw_op(
                    block,
                    "waveActiveBit.i32",
                    Attr::NoUnwind,
                    ret,
                    120,
                    &[low_arg, args[1]],
                );
                let high = self.raw_op(
                    block,
                    "waveActiveBit.i32",
                    Attr::NoUnwind,
                    ret,
                    120,
                    &[high_arg, args[1]],
                );
                self.m.select(block, upper, high, low)
            }
            _ => unreachable!(),
        }
    }

    fn raw_op(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        #[cfg(test)]
        self.opcodes.push(opcode);
        let mut params = vec![self.t.i32];
        params.extend(args.iter().map(|&a| self.m.type_of(a)));
        let fty = self.m.func_ty(ret, params);
        self.m.ptr(fty);
        let f = self
            .m
            .function(&format!("dx.op.{name}"), fty, Some(attr), false);
        let mut all = vec![self.m.i32(opcode)];
        all.extend(args);
        self.m.call(block, f, all)
    }

    fn allocate_range(&mut self, class: DescriptorClass) -> u32 {
        let slot = &mut self.next_ranges[class.index()];
        let id = *slot;
        *slot += 1;
        id
    }

    fn create_handle(
        &mut self,
        block: usize,
        class: DescriptorClass,
        range: u32,
        index: V,
        nonuniform: bool,
        bind: V,
    ) -> V {
        if self.profile == super::ShaderProfile::Sm65 {
            let class = self.m.i8(class.code());
            let range = self.m.i32(range);
            let nonuniform = self.m.bool(nonuniform);
            self.op(
                block,
                "createHandle",
                Attr::ReadOnly,
                self.t.handle,
                57,
                &[class, range, index, nonuniform],
            )
        } else {
            let nonuniform = self.m.bool(nonuniform);
            self.op(
                block,
                "createHandleFromBinding",
                Attr::ReadNone,
                self.t.handle,
                217,
                &[bind, index, nonuniform],
            )
        }
    }

    /// The handle of the user-data constant buffer, made in the entry block.
    pub fn user_data_handle(&mut self) -> V {
        let bind = [
            self.m.i32(0),
            self.m.i32(0),
            self.m.i32(USER_DATA_SPACE),
            self.m.i8(2), // CBV
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let i = self.m.i32(0);
        let range = self.allocate_range(DescriptorClass::Cbv);
        self.user_data_range = Some(range);
        let h = self.create_handle(block, DescriptorClass::Cbv, range, i, false, bind);
        // The size is known in `finish`, which fixes this placeholder.
        self.annotate(block, h, 13, u32::MAX)
    }

    /// The handle of the root CBV of root parameter `slot`, made in the
    /// entry block.
    pub fn root_cbv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_cbvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(2)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let range = self.allocate_range(DescriptorClass::Cbv);
        let h = self.create_handle(block, DescriptorClass::Cbv, range, r, false, bind);
        let h = self.annotate(block, h, 13, 65536);
        self.root_cbvs.push((slot, h));
        self.root_cbv_ranges.push(range);
        h
    }

    /// The handle of the root SRV (a raw buffer) of root parameter `slot`.
    pub fn root_srv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_srvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let range = self.allocate_range(DescriptorClass::Srv);
        let h = self.create_handle(block, DescriptorClass::Srv, range, r, false, bind);
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_srvs.push((slot, h));
        self.root_srv_ranges.push(range);
        h
    }

    /// The handle of the root UAV (a raw buffer) of root parameter `slot`:
    /// `u<slot>` in [`ROOT_CBV_SPACE`].
    pub fn root_uav_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_uavs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let range = self.allocate_range(DescriptorClass::Uav);
        let h = self.create_handle(block, DescriptorClass::Uav, range, r, false, bind);
        let h = self.annotate(block, h, 11 | 0x1000, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_uavs.push((slot, h));
        self.root_uav_ranges.push(range);
        h
    }

    /// The handle of immediate sampler `n` (a comparison sampler when its
    /// bit in `cmp_samplers` is set).
    pub fn sampler_handle(&mut self, block: usize, n: u32) -> V {
        let bind = [
            self.m.i32(n),
            self.m.i32(n),
            self.m.i32(SAMPLER_SPACE),
            self.m.i8(3), // sampler
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let i = self.m.i32(n);
        if self.sampler_ranges.len() <= n as usize {
            self.sampler_ranges.resize(n as usize + 1, u32::MAX);
        }
        if self.sampler_ranges[n as usize] == u32::MAX {
            self.sampler_ranges[n as usize] = self.allocate_range(DescriptorClass::Sampler);
        }
        let range = self.sampler_ranges[n as usize];
        let h = self.create_handle(block, DescriptorClass::Sampler, range, i, false, bind);
        let cmp = if self.cmp_samplers >> n & 1 != 0 {
            0x8000
        } else {
            0
        };
        self.annotate(block, h, 14 | cmp, 0)
    }

    pub fn annotate(&mut self, block: usize, h: V, props0: u32, props1: u32) -> V {
        if self.profile == super::ShaderProfile::Sm65 {
            return h;
        }
        let p = [self.m.i32(props0), self.m.i32(props1)];
        let p = self.m.aggregate(self.t.props, &p);
        self.op(
            block,
            "annotateHandle",
            Attr::ReadNone,
            self.t.handle,
            216,
            &[h, p],
        )
    }

    /// A descriptor heap handle, annotated with `props`; `nonuniform` when
    /// the index may differ between lanes (`NonUniformResourceIndex`).
    pub fn heap_handle(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
    ) -> V {
        let nonuniform = nonuniform || self.half32;
        // a view index past the host heap (`set_view_heap_limit`) reads its null slot instead
        let limit = VIEW_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
        // (as `dx.op.binary` UMin: a compare and select emitted here made the bitcode invalid)
        let index = if !sampler && limit != 0 {
            let null = self.m.i32(limit);
            let i32t = self.t.i32;
            let v = self.op(
                block,
                "binary.i32",
                Attr::ReadNone,
                i32t,
                40,
                &[index, null],
            );
            self.m.tag(v, super::bitcode::Tag::Clamp);
            v
        } else if sampler {
            // A sampler heap has at most 2,048 samplers on any host: an
            // index past that (a number read from memory the host has not
            // got yet) stays inside the largest heap there is.
            // shortcut: not the bound heap's own count, which a title
            // heap smaller than 2,048 would need.
            let last = self.m.i32(2047);
            let i32t = self.t.i32;
            let v = self.op(
                block,
                "binary.i32",
                Attr::ReadNone,
                i32t,
                40,
                &[index, last],
            );
            self.m.tag(v, super::bitcode::Tag::Clamp);
            v
        } else {
            index
        };
        self.nonuniform_handles += u32::from(nonuniform);
        if self.profile == super::ShaderProfile::Sm65 {
            let key = BindingKey::from_props(sampler, props);
            if matches!(key.kind, 11 | 12) {
                self.flags |= flags::RAW_BUFFERS;
            }
            let table = match self.descriptor_tables.iter().find(|t| t.key == key) {
                Some(table) => *table,
                None => {
                    let table = DescriptorTable {
                        key,
                        space: DYNAMIC_SPACE_BASE + self.descriptor_tables.len() as u32,
                        range_id: self.allocate_range(key.class),
                        upper: u32::MAX,
                    };
                    self.descriptor_tables.push(table);
                    table
                }
            };
            let h = self.create_handle(block, key.class, table.range_id, index, nonuniform, index);
            return self.annotate(block, h, props.0, props.1);
        }
        self.flags |= if sampler {
            flags::SAMPLER_HEAP
        } else {
            flags::RESOURCE_HEAP
        };
        let (s, n) = (self.m.bool(sampler), self.m.bool(nonuniform));
        let h = self.op(
            block,
            "createHandleFromHeap",
            Attr::ReadNone,
            self.t.handle,
            218,
            &[index, s, n],
        );
        self.annotate(block, h, props.0, props.1)
    }

    /// The declared type of a dynamic-heap alias: what the DXIL metadata
    /// and the validator read its shape and element properties from.
    fn alias_type(&mut self, key: BindingKey) -> Ty {
        let scalar = if key.component & 0xff == 9 {
            self.t.f32
        } else {
            self.t.i32
        };
        let four = self.m.vector(4, scalar);
        let ty = match key.kind {
            11 => {
                let name = if key.class == DescriptorClass::Uav {
                    "struct.RWByteAddressBuffer"
                } else {
                    "struct.ByteAddressBuffer"
                };
                self.m.named_struct(name, vec![self.t.i32])
            }
            12 => {
                let stride = key.stride.max(4);
                let mut fields = vec![self.m.array(stride / 4, self.t.i32)];
                let rest = stride % 4;
                if rest != 0 {
                    let i8t = self.m.int(8);
                    fields.push(self.m.array(rest, i8t));
                }
                let elem = self
                    .m
                    .named_struct(&format!("struct.victory.sb{stride}"), fields);
                let name = format!(
                    "class.{}StructuredBuffer<victory.sb{stride}>",
                    if key.class == DescriptorClass::Uav {
                        "RW"
                    } else {
                        ""
                    }
                );
                self.m.named_struct(&name, vec![elem])
            }
            10 => {
                let name = format!(
                    "class.{}Buffer<victory.v4.c{}>",
                    if key.class == DescriptorClass::Uav {
                        "RW"
                    } else {
                        ""
                    },
                    key.component & 0xff
                );
                self.m.named_struct(&name, vec![four])
            }
            13 => {
                let bytes = key.stride.max(16);
                let mut fields = vec![self.m.array(bytes / 4, self.t.i32)];
                let rest = bytes % 4;
                if rest != 0 {
                    let i8t = self.m.int(8);
                    fields.push(self.m.array(rest, i8t));
                }
                self.m
                    .named_struct(&format!("struct.victory.cbv{bytes}"), fields)
            }
            14 => self.m.named_struct("struct.SamplerState", vec![self.t.i32]),
            1..=9 => {
                let dim = match key.kind {
                    1 => "Texture1D",
                    2 => "Texture2D",
                    3 => "Texture2DMS",
                    4 => "Texture3D",
                    5 => "TextureCube",
                    6 => "Texture1DArray",
                    7 => "Texture2DArray",
                    8 => "Texture2DMSArray",
                    _ => "TextureCubeArray",
                };
                let base = format!("class.{dim}<victory.v4.c{}>", key.component & 0xff);
                let mips = self
                    .m
                    .named_struct(&format!("{base}::mips_type"), vec![self.t.i32]);
                self.m.named_struct(&base, vec![four, mips])
            }
            _ => self
                .m
                .named_struct("struct.ByteAddressBuffer", vec![self.t.i32]),
        };
        // An unbounded range's variable is an array of no fixed length
        // (a cbuffer array says `4294967295`, as DXC emits it).
        let elements = if key.class == DescriptorClass::Cbv {
            u32::MAX
        } else {
            0
        };
        self.m.array(elements, ty)
    }
    /// Whether the program makes handles from a descriptor heap directly
    /// (SM 6.6 `ResourceDescriptorHeap[]`/`SamplerDescriptorHeap[]`).
    pub fn direct_heap(&self) -> bool {
        self.flags & (flags::RESOURCE_HEAP | flags::SAMPLER_HEAP) != 0
    }

    /// The finished container.
    pub fn finish(self) -> Vec<u8> {
        self.try_finish()
            .expect("shader composites must be projected")
    }

    pub fn try_finish(mut self) -> Result<Vec<u8>, &'static str> {
        self.m.validate_composites()?;
        super::bitcode::input_signature::compact_software_barycentrics(&mut self)?;
        Ok(self.finish_with_tags_validated().0)
    }

    /// The finished container and the entry-point property tags it
    /// declares (the profile tests read them).
    #[cfg(test)]
    pub fn finish_with_tags(mut self) -> (Vec<u8>, Vec<u32>) {
        self.m
            .validate_composites()
            .expect("shader composites must be projected");
        super::bitcode::input_signature::compact_software_barycentrics(&mut self)
            .expect("software barycentric input signature must fit");
        self.finish_with_tags_validated()
    }

    fn finish_with_tags_validated(mut self) -> (Vec<u8>, Vec<u32>) {
        let physical_wave = if self.half32 {
            Some(64)
        } else {
            physical_wave_size(self.wave_size, self.threads, super::lower::host_min_wave())
        };
        let ud_size = (self.user_data * 4).next_multiple_of(16);
        self.fix_user_data_size(ud_size);
        // The clean-up, before anything is declared about the code: a
        // feature whose last use it removed must not be declared (the
        // validator compares the flags with what the code uses).
        let passes = super::shader_opt() & super::bitcode::opt::MODULE;
        if passes != 0 {
            self.m.optimise(passes);
            let used = self.m.usage();
            for (flag, used) in [
                (flags::WAVE_OPS, used.wave_ops),
                (flags::INT64, used.int64),
                (flags::TYPED_UAV_LOADS, used.typed_uav_loads),
                (flags::RESOURCE_HEAP, used.heap_views),
                (flags::SAMPLER_HEAP, used.heap_samplers),
                (flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64, used.atomic64),
            ] {
                if !used {
                    self.flags &= !flag;
                }
            }
        }
        let mut resources = None;
        let mut bindings = Vec::new();
        let i32c = |s: &mut Self, v: u32| {
            let c = s.m.i32(v);
            s.m.md_value(c)
        };
        if self.user_data > 0
            || self.samplers > 0
            || !self.root_cbvs.is_empty()
            || !self.root_srvs.is_empty()
            || !self.root_uavs.is_empty()
            || !self.descriptor_tables.is_empty()
        {
            let empty = self.m.md_str("");
            let no = self.m.bool(false);
            let no = self.m.md_value(no);
            let undef_of = |s: &mut Self, ty: Ty| {
                let p = s.m.ptr(ty);
                let u = s.m.undef(p);
                s.m.md_value(u)
            };
            // Every range in a class, keyed by the class-local id `createHandle`
            // names; the lists are written sorted, so an id is its position.
            let mut made: [Vec<Made>; 4] = std::array::from_fn(|_| Vec::new());
            let mut add = |class: DescriptorClass, range: u32, node: Md, binding: Binding| {
                made[class.index()].push(Made {
                    range,
                    node,
                    binding,
                });
            };
            if self.user_data > 0 {
                let arr = self.m.array(self.user_data.next_multiple_of(4), self.t.i32);
                let ty = self.m.named_struct("UserData", vec![arr]);
                let var = undef_of(&mut self, ty);
                let range = self.user_data_range.unwrap_or(0);
                let ops = vec![
                    Some(i32c(&mut self, range)),
                    Some(var),
                    Some(empty),
                    Some(i32c(&mut self, USER_DATA_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, ud_size)),
                    None,
                ];
                add(
                    DescriptorClass::Cbv,
                    range,
                    self.m.md_node(ops),
                    Binding {
                        kind: 2,
                        space: USER_DATA_SPACE,
                        register: 0,
                        upper: 0,
                        dxil_kind: 13,
                    },
                );
            }
            let blob = self.m.array(16384, self.t.i32);
            let root_cbv = self.m.named_struct("RootCbv", vec![blob]);
            let root_cbv_var = undef_of(&mut self, root_cbv);
            for ((slot, _), range) in self
                .root_cbvs
                .clone()
                .into_iter()
                .zip(self.root_cbv_ranges.clone())
            {
                let ops = vec![
                    Some(i32c(&mut self, range)),
                    Some(root_cbv_var),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 65536)),
                    None,
                ];
                add(
                    DescriptorClass::Cbv,
                    range,
                    self.m.md_node(ops),
                    Binding {
                        kind: 2,
                        space: ROOT_CBV_SPACE,
                        register: u32::from(slot),
                        upper: u32::from(slot),
                        dxil_kind: 13,
                    },
                );
            }
            let raw_srv = self
                .m
                .named_struct("struct.ByteAddressBuffer", vec![self.t.i32]);
            let raw_srv_var = undef_of(&mut self, raw_srv);
            for ((slot, _), range) in self
                .root_srvs
                .clone()
                .into_iter()
                .zip(self.root_srv_ranges.clone())
            {
                let ops = vec![
                    Some(i32c(&mut self, range)),
                    Some(raw_srv_var),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 11)),
                    Some(i32c(&mut self, 0)),
                    None,
                ];
                add(
                    DescriptorClass::Srv,
                    range,
                    self.m.md_node(ops),
                    Binding {
                        kind: 4,
                        space: ROOT_CBV_SPACE,
                        register: u32::from(slot),
                        upper: u32::from(slot),
                        dxil_kind: 11,
                    },
                );
            }
            let raw_uav = self
                .m
                .named_struct("struct.RWByteAddressBuffer", vec![self.t.i32]);
            let raw_uav_var = undef_of(&mut self, raw_uav);
            for ((slot, _), range) in self
                .root_uavs
                .clone()
                .into_iter()
                .zip(self.root_uav_ranges.clone())
            {
                // Id, variable, name, space, lower bound, range size,
                // shape (11: raw buffer), globally coherent, counter,
                // rasterizer ordered, no extra properties.
                let ops = vec![
                    Some(i32c(&mut self, range)),
                    Some(raw_uav_var),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 11)),
                    Some(no),
                    Some(no),
                    Some(no),
                    None,
                ];
                add(
                    DescriptorClass::Uav,
                    range,
                    self.m.md_node(ops),
                    Binding {
                        kind: 7,
                        space: ROOT_CBV_SPACE,
                        register: u32::from(slot),
                        upper: u32::from(slot),
                        dxil_kind: 11,
                    },
                );
            }
            let sampler_ty = self.m.named_struct("struct.SamplerState", vec![self.t.i32]);
            let sampler_var = undef_of(&mut self, sampler_ty);
            let mut next_unused = self.next_ranges[DescriptorClass::Sampler.index()];
            for n in 0..self.samplers {
                let used = self
                    .sampler_ranges
                    .get(n as usize)
                    .copied()
                    .filter(|&r| r != u32::MAX);
                let range = used.unwrap_or_else(|| {
                    let r = next_unused;
                    next_unused += 1;
                    r
                });
                if used.is_none() {
                    self.next_ranges[DescriptorClass::Sampler.index()] = next_unused;
                }
                let kind = (self.cmp_samplers >> n & 1) as u32;
                let ops = vec![
                    Some(i32c(&mut self, range)),
                    Some(sampler_var),
                    Some(empty),
                    Some(i32c(&mut self, SAMPLER_SPACE)),
                    Some(i32c(&mut self, n)),
                    Some(i32c(&mut self, 1)),
                    // Sampler kind: 1 comparison.
                    Some(i32c(&mut self, kind)),
                    None,
                ];
                add(
                    DescriptorClass::Sampler,
                    range,
                    self.m.md_node(ops),
                    Binding {
                        kind: 1,
                        space: SAMPLER_SPACE,
                        register: n,
                        upper: n,
                        dxil_kind: 14,
                    },
                );
            }
            for table in self.descriptor_tables.clone() {
                let key = table.key;
                let range_size = if table.upper == u32::MAX {
                    u32::MAX
                } else {
                    table.upper + 1
                };
                let var = {
                    let ty = self.alias_type(key);
                    undef_of(&mut self, ty)
                };
                let shape = key.kind;
                let extra = match key.kind {
                    11 | 13 | 14 => None,
                    12 => {
                        let one = Some(i32c(&mut self, 1));
                        let stride = Some(i32c(&mut self, key.stride));
                        Some(self.m.md_node(vec![one, stride]))
                    }
                    _ => {
                        let zero = Some(i32c(&mut self, 0));
                        let comp = Some(i32c(&mut self, key.component & 0xff));
                        Some(self.m.md_node(vec![zero, comp]))
                    }
                };
                let (node, binding) = match key.class {
                    DescriptorClass::Cbv => {
                        let ops = vec![
                            Some(i32c(&mut self, table.range_id)),
                            Some(var),
                            Some(empty),
                            Some(i32c(&mut self, table.space)),
                            Some(i32c(&mut self, 0)),
                            Some(i32c(&mut self, range_size)),
                            Some(i32c(&mut self, key.stride.max(16))),
                            None,
                        ];
                        (
                            self.m.md_node(ops),
                            Binding {
                                kind: 2,
                                space: table.space,
                                register: 0,
                                upper: table.upper,
                                dxil_kind: 13,
                            },
                        )
                    }
                    DescriptorClass::Srv => {
                        let ops = vec![
                            Some(i32c(&mut self, table.range_id)),
                            Some(var),
                            Some(empty),
                            Some(i32c(&mut self, table.space)),
                            Some(i32c(&mut self, 0)),
                            Some(i32c(&mut self, range_size)),
                            Some(i32c(&mut self, shape)),
                            Some(i32c(&mut self, 0)),
                            extra,
                        ];
                        (
                            self.m.md_node(ops),
                            Binding {
                                kind: psv_resource_type(key),
                                space: table.space,
                                register: 0,
                                upper: table.upper,
                                dxil_kind: shape,
                            },
                        )
                    }
                    DescriptorClass::Uav => {
                        let counter = if key.counter {
                            let b = self.m.bool(true);
                            self.m.md_value(b)
                        } else {
                            no
                        };
                        let ops = vec![
                            Some(i32c(&mut self, table.range_id)),
                            Some(var),
                            Some(empty),
                            Some(i32c(&mut self, table.space)),
                            Some(i32c(&mut self, 0)),
                            Some(i32c(&mut self, range_size)),
                            Some(i32c(&mut self, shape)),
                            Some(no),
                            Some(counter),
                            Some(no),
                            extra,
                        ];
                        (
                            self.m.md_node(ops),
                            Binding {
                                kind: psv_resource_type(key),
                                space: table.space,
                                register: 0,
                                upper: table.upper,
                                dxil_kind: shape,
                            },
                        )
                    }
                    DescriptorClass::Sampler => {
                        let ops = vec![
                            Some(i32c(&mut self, table.range_id)),
                            Some(var),
                            Some(empty),
                            Some(i32c(&mut self, table.space)),
                            Some(i32c(&mut self, 0)),
                            Some(i32c(&mut self, range_size)),
                            Some(i32c(&mut self, 0)),
                            None,
                        ];
                        (
                            self.m.md_node(ops),
                            Binding {
                                kind: 1,
                                space: table.space,
                                register: 0,
                                upper: table.upper,
                                dxil_kind: 14,
                            },
                        )
                    }
                };
                add(key.class, table.range_id, node, binding);
            }
            let mut lists = [None; 4];
            for (class, mut list) in made.into_iter().enumerate() {
                if list.is_empty() {
                    continue;
                }
                list.sort_by_key(|m| m.range);
                for (id, m) in list.iter().enumerate() {
                    debug_assert_eq!(m.range, id as u32);
                    bindings.push(m.binding);
                    if id == list.len() - 1 {
                        lists[class] =
                            Some(self.m.md_node(list.iter().map(|m| Some(m.node)).collect()));
                    }
                }
            }
            resources = Some(self.m.md_node(vec![
                lists[DescriptorClass::Srv.index()],
                lists[DescriptorClass::Uav.index()],
                lists[DescriptorClass::Cbv.index()],
                lists[DescriptorClass::Sampler.index()],
            ]));
            // PSV0 lists them as the module does: CBVs, samplers, SRVs, UAVs.
            bindings.sort_by_key(|b| match b.kind {
                2 => 0,
                1 => 1,
                6..=9 => 3,
                _ => 2,
            });
        }
        if let Some(r) = resources {
            self.m.named_md("dx.resources", vec![r]);
        }

        let graphics = self.threads.is_none() || self.mesh.is_some();
        if graphics {
            // shortcut: every output is declared to depend on every input; the
            // exact sets need a dependence pass. Drivers read this state.
            let scalars = |els: &[SigElement]| {
                els.iter()
                    .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                    .max()
                    .unwrap_or(0)
            };
            let (ni, no) = (scalars(&self.inputs), scalars(&self.outputs));
            let mut view = vec![ni, no];
            // Mesh shaders have no inputs: then the primitive outputs' count.
            if self.mesh.is_some() {
                view.push(scalars(&self.prims));
            }
            for _ in 0..ni {
                for k in 0..no.div_ceil(32) {
                    let bits = (no - 32 * k).min(32);
                    view.push(if bits == 32 {
                        u32::MAX
                    } else {
                        (1 << bits) - 1
                    });
                }
            }
            // A geometry shader lists four streams: the other three have
            // no outputs.
            if self.geometry.is_some() {
                view.extend([0, 0, 0]);
            }
            let arr = self.m.array(view.len() as u32, self.t.i32);
            let elems: Vec<V> = view.iter().map(|&v| self.m.i32(v)).collect();
            let arr = self.m.aggregate(arr, &elems);
            let arr = self.m.md_value(arr);
            let view_node = self.m.md_node(vec![Some(arr)]);
            self.m.named_md("dx.viewIdState", vec![view_node]);
        }

        let sigs = if graphics {
            let ins = self.signature_md(self.inputs.clone());
            let outs = self.signature_md(self.outputs.clone());
            let prims = self.signature_md(self.prims.clone());
            Some(self.m.md_node(vec![ins, outs, prims]))
        } else {
            None
        };
        // A pixel shader with an SV_StencilRef output (as DXC collects it).
        if self.pixel && self.outputs.iter().any(|o| o.kind == 20) {
            self.flags |= flags::STENCIL_REF;
        }
        // Entry properties: shader flags (tag 0), geometry state (1),
        // thread-group size (4), wave size (11).
        let mut props = Vec::new();
        let mut property_tags = Vec::new();
        if self.flags != 0 {
            property_tags.push(0);
            props.push(i32c(&mut self, 0));
            let i64t = self.m.int(64);
            let f = self.m.int_const(i64t, self.flags as i64);
            props.push(self.m.md_value(f));
        }
        if let Some(g) = self.geometry {
            // Input primitive (3: triangles), maximum vertices, stream
            // mask, output topology (5: triangle strips), instances.
            property_tags.push(1);
            props.push(i32c(&mut self, 1));
            let state = [3, g.max_vertices, 1, 5, 1].map(|n| Some(i32c(&mut self, n)));
            props.push(self.m.md_node(state.to_vec()));
        }
        if let Some(threads) = self.threads {
            let t = threads.map(|n| Some(i32c(&mut self, n)));
            let t = self.m.md_node(t.to_vec());
            match self.mesh {
                // Mesh state (tag 9): threads, maximum vertices and
                // primitives, topology, payload bytes.
                Some(mesh) => {
                    property_tags.push(9);
                    props.push(i32c(&mut self, 9));
                    let state = [
                        mesh.max_vertices,
                        mesh.max_primitives,
                        mesh.topology.into(),
                        0,
                    ]
                    .map(|n| Some(i32c(&mut self, n)));
                    let mut node = vec![Some(t)];
                    node.extend(state);
                    props.push(self.m.md_node(node));
                }
                None => {
                    property_tags.push(4);
                    props.push(i32c(&mut self, 4));
                    props.push(t);
                }
            }
        }
        if let Some(n) = physical_wave
            && self.profile == super::ShaderProfile::Sm66
        {
            property_tags.push(11);
            props.push(i32c(&mut self, 11));
            let n = Some(i32c(&mut self, n));
            props.push(self.m.md_node(vec![n]));
        }
        let props =
            (!props.is_empty()).then(|| self.m.md_node(props.into_iter().map(Some).collect()));
        let main = self.m.md_value(self.main);
        let name = self.m.md_str("main");
        let entry = self
            .m
            .md_node(vec![Some(main), Some(name), sigs, resources, props]);

        let ident = self.m.md_str("victory");
        let ident = self.m.md_node(vec![Some(ident)]);
        self.m.named_md("llvm.ident", vec![ident]);
        let one = i32c(&mut self, 1);
        let six = i32c(&mut self, 6);
        let validator_version = self.profile.validator_version();
        let validator = i32c(&mut self, validator_version);
        let version = self.m.md_node(vec![Some(one), Some(validator)]);
        self.m.named_md("dx.version", vec![version]);
        self.m.named_md("dx.valver", vec![version]);
        // Program kinds: 0 pixel, 1 vertex, 2 geometry, 5 compute, 13 mesh.
        let (name, kind) = match (self.threads, self.pixel) {
            _ if self.mesh.is_some() => ("ms", 13u32),
            _ if self.geometry.is_some() => ("gs", 2),
            (Some(_), _) => ("cs", 5u32),
            (None, true) => ("ps", 0),
            (None, false) => ("vs", 1),
        };
        let name = self.m.md_str(name);
        let sm_minor = u32::from(self.profile.shader_model().1);
        let minor = i32c(&mut self, sm_minor);
        let sm = self.m.md_node(vec![Some(name), Some(six), Some(minor)]);
        self.m.named_md("dx.shaderModel", vec![sm]);
        self.m.named_md("dx.entryPoints", vec![entry]);

        let bitcode = self.m.write();
        super::bitcode::census::note(&self.m, &bitcode);
        let mut dxil = Vec::with_capacity(24 + bitcode.len());
        let (major, minor) = self.profile.shader_model();
        let program = kind << 16 | u32::from(major) << 4 | u32::from(minor);
        dxil.extend(program.to_le_bytes());
        dxil.extend((((24 + bitcode.len()) / 4) as u32).to_le_bytes());
        dxil.extend(b"DXIL");
        dxil.extend(self.profile.dxil_version().to_le_bytes());
        dxil.extend(16u32.to_le_bytes());
        dxil.extend((bitcode.len() as u32).to_le_bytes());
        dxil.extend(&bitcode);

        let mut features = 0u64;
        if self.flags & flags::STENCIL_REF != 0 {
            features |= 0x200;
        }
        if self.flags & flags::TYPED_UAV_LOADS != 0 {
            features |= 0x800;
        }
        if self.flags & flags::WAVE_OPS != 0 {
            features |= 0x4000;
        }
        if self.flags & flags::INT64 != 0 {
            features |= 0x8000;
        }
        if self.flags & flags::BARYCENTRICS != 0 {
            features |= 0x2_0000;
        }
        if self.profile == super::ShaderProfile::Sm66 {
            if self.flags & flags::RESOURCE_HEAP != 0 {
                features |= 0x200_0000;
            }
            if self.flags & flags::SAMPLER_HEAP != 0 {
                features |= 0x400_0000;
            }
            if self.flags & flags::ATOMIC_TYPED64 != 0 {
                features |= 0x40_0000;
            }
            if self.flags & flags::ATOMIC_HEAP64 != 0 {
                features |= 0x1000_0000;
            }
        }
        let sfi0 = features.to_le_bytes();
        let empty = Vec::new();
        let (sig_in, sig_out, sig_prims) = if graphics {
            (&self.inputs, &self.outputs, &self.prims)
        } else {
            (&empty, &empty, &empty)
        };
        let isg1 = container::signature(sig_in, true);
        let osg1 = container::signature(sig_out, false);
        let flag = if self.pixel {
            sig_out.iter().any(|o| (17..=19).contains(&o.kind))
        } else {
            sig_out.iter().any(|o| o.kind == 3)
        };
        let psv0 = container::psv(
            &Psv {
                stage: kind as u8,
                flag: flag && self.threads.is_none(),
                inputs: sig_in,
                outputs: sig_out,
                prims: sig_prims,
                mesh: self.mesh.map(|m| (m, self.lds_dwords * 4)),
                geometry: self.geometry.map(|g| g.max_vertices),
                bindings: &bindings,
                threads: self.threads.unwrap_or_default(),
                // SM 6.5 has no `WaveSize` property: the runtime info says
                // no wave size is required even when the translation needs
                // 64-lane waves of the console.
                wave_size: if self.profile == super::ShaderProfile::Sm66 {
                    physical_wave
                } else {
                    None
                },
            },
            self.profile,
        );
        let hash = [0u8; 20];
        let psg1 = container::signature(sig_prims, false);
        let mut parts: Vec<(&[u8; 4], &[u8])> =
            vec![(b"SFI0", &sfi0), (b"ISG1", &isg1), (b"OSG1", &osg1)];
        if self.mesh.is_some() {
            parts.push((b"PSG1", &psg1));
        }
        parts.extend([(b"PSV0", &psv0[..]), (b"HASH", &hash), (b"DXIL", &dxil)]);
        let container = container::build(&parts);
        self.property_tags = property_tags;
        (container, self.property_tags)
    }

    /// The user-data buffer's annotation carries its size, known only now.
    fn fix_user_data_size(&mut self, size: u32) {
        let old = [self.m.i32(13), self.m.i32(u32::MAX)];
        let old = self.m.aggregate(self.t.props, &old);
        let new = [self.m.i32(13), self.m.i32(size)];
        let new = self.m.aggregate(self.t.props, &new);
        self.m.replace_const(old, new);
    }

    fn signature_md(&mut self, elements: Vec<SigElement>) -> Option<Md> {
        if elements.is_empty() {
            return None;
        }
        let mut list = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let c32 = |s: &mut Self, v: u32| {
                let c = s.m.i32(v);
                Some(s.m.md_value(c))
            };
            let c8 = |s: &mut Self, v: u8| {
                let c = s.m.i8(v);
                Some(s.m.md_value(c))
            };
            let idx = c32(self, e.index);
            let indices = Some(self.m.md_node(vec![idx]));
            let usage = if e.usage != 0 {
                let (tag, mask) = (c32(self, 3), c32(self, u32::from(e.usage)));
                Some(self.m.md_node(vec![tag, mask]))
            } else {
                None
            };
            let ops = vec![
                c32(self, id as u32),
                Some(self.m.md_str(&e.name)),
                c8(self, e.comp),
                c8(self, e.kind),
                indices,
                c8(self, e.interp),
                c32(self, 1),
                c8(self, e.cols),
                c32(self, e.row.unwrap_or(u32::MAX)),
                c8(self, if e.row.is_some() { e.col } else { u8::MAX }),
                usage,
            ];
            list.push(Some(self.m.md_node(ops)));
        }
        Some(self.m.md_node(list))
    }
}
