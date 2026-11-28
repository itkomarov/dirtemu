use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak, mpsc};

use super::host::GraphicsState;
use super::make::{self, Pending};

static DIR: OnceLock<PathBuf> = OnceLock::new();
static PREPARED: OnceLock<Mutex<HashMap<String, Prepared>>> = OnceLock::new();
static REUSED: AtomicU64 = AtomicU64::new(0);
static FILE_ID: AtomicU64 = AtomicU64::new(0);

const STAGES: [&str; 8] = [
    "compute",
    "vertex",
    "pixel",
    "mesh",
    "amplification",
    "geometry",
    "domain",
    "hull",
];
const MAX_SHADER: u64 = 16 << 20;
const MAX_RECIPE: u64 = 64 << 10;

pub(super) fn configure(dir: PathBuf) -> io::Result<()> {
    std::fs::create_dir_all(dir.join("shaders"))?;
    std::fs::create_dir_all(dir.join("states"))?;
    let _ = DIR.set(dir);
    Ok(())
}

pub(super) fn take(key: &str) -> Option<Arc<Pending>> {
    let mut cache = PREPARED.get()?.lock().unwrap_or_else(|e| e.into_inner());
    let state = cache.get_mut(key)?;
    let pipeline = state.take()?;
    let reused = REUSED.fetch_add(1, Ordering::Relaxed) + 1;
    if reused.is_power_of_two() {
        tracing::info!(
            reused,
            "precompiled pipeline states matched at title creation"
        );
    }
    Some(pipeline)
}

struct Prepared {
    waiting: Option<Arc<Pending>>,
    active: Weak<Pending>,
}

impl Prepared {
    fn take(&mut self) -> Option<Arc<Pending>> {
        self.waiting.take().or_else(|| self.active.upgrade())
    }
}

enum Work {
    Save(Arc<Pending>),
    Flush(mpsc::Sender<()>),
}

fn recorder() -> Option<&'static mpsc::Sender<Work>> {
    static WRITER: OnceLock<Option<mpsc::Sender<Work>>> = OnceLock::new();
    let dir = DIR.get()?;
    WRITER.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        let result = std::thread::Builder::new().name("victory-pipeline-recipes".into()).spawn(move || {
            for work in rx {
                match work {
                    Work::Save(pipeline) => {
                        if let Err(error) = save(dir, &pipeline) {
                            tracing::warn!(%error, key = pipeline.key(), "pipeline recipe not saved");
                        }
                    }
                    Work::Flush(done) => { let _ = done.send(()); }
                }
            }
        });
        match result {
            Ok(_) => Some(tx),
            Err(error) => {
                tracing::warn!(%error, "pipeline recipe writer could not start");
                None
            }
        }
    }).as_ref()
}

pub(super) fn record(p: &Arc<Pending>) {
    if p.shaders().is_empty() {
        return;
    }
    if let Some(writer) = recorder() {
        static RECORDED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
        let mut recorded = RECORDED.lock().unwrap_or_else(|e| e.into_inner());
        let recorded = recorded.get_or_insert_with(HashSet::new);
        if !recorded.contains(p.key()) && writer.send(Work::Save(p.clone())).is_ok() {
            recorded.insert(p.key().to_owned());
        }
    }
}

pub(super) fn flush() {
    if let Some(writer) = recorder() {
        let (tx, rx) = mpsc::channel();
        if writer.send(Work::Flush(tx)).is_ok() {
            let _ = rx.recv();
        }
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if path.exists() {
        return Ok(());
    }
    let serial = FILE_ID.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("{}.{}.tmp", std::process::id(), serial));
    let result = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    if path.exists() { Ok(()) } else { result }
}

fn save(dir: &Path, pending: &Pending) -> io::Result<()> {
    let state = dir.join("states").join(pending.key());
    let shaders = pending.shaders();
    if shaders.is_empty() || shaders.len() > STAGES.len() || state.exists() {
        return Ok(());
    }
    let mut bytes = b"VPR1".to_vec();
    bytes.push(shaders.len() as u8);
    for shader in shaders {
        let Some(stage) = STAGES.iter().position(|stage| *stage == shader.stage) else {
            return Ok(());
        };
        if shader.container.len() as u64 > MAX_SHADER {
            return Ok(());
        }
        bytes.push(stage as u8);
        let hash = blake3::hash(&shader.container);
        write_new(
            &dir.join("shaders").join(hash.to_hex().as_str()),
            &shader.container,
        )?;
        bytes.extend(hash.as_bytes());
    }
    if let Some(graphics) = &pending.graphics {
        bytes.extend(graphics.encode_recipe());
    }
    if bytes.len() as u64 > MAX_RECIPE {
        return Ok(());
    }
    write_new(&state, &bytes)
}

fn small_file(path: &Path, max: u64) -> io::Result<Vec<u8>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(max + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err(io::Error::other("cache file is too large"));
    }
    Ok(bytes)
}

fn load(dir: &Path, path: &Path) -> anyhow::Result<Arc<Pending>> {
    let bytes = small_file(path, MAX_RECIPE)?;
    let bytes = bytes
        .strip_prefix(b"VPR1")
        .ok_or_else(|| anyhow::anyhow!("invalid recipe header"))?;
    let (&count, bytes) = bytes
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("missing shader count"))?;
    anyhow::ensure!(
        count != 0 && usize::from(count) <= STAGES.len(),
        "invalid shader count"
    );
    let (stages, graphics) = bytes
        .split_at_checked(usize::from(count) * 33)
        .ok_or_else(|| anyhow::anyhow!("truncated recipe"))?;
    let graphics = if graphics.is_empty() {
        None
    } else {
        Some(GraphicsState::decode_recipe(graphics).map_err(anyhow::Error::msg)?)
    };
    let mut shaders = Vec::with_capacity(usize::from(count));
    for record in stages.chunks_exact(33) {
        let stage = *STAGES
            .get(usize::from(record[0]))
            .ok_or_else(|| anyhow::anyhow!("invalid shader stage"))?;
        let hash = blake3::Hash::from_bytes(record[1..].try_into()?);
        let container = small_file(
            &dir.join("shaders").join(hash.to_hex().as_str()),
            MAX_SHADER,
        )?;
        anyhow::ensure!(
            blake3::hash(&container) == hash,
            "shader does not match recipe"
        );
        shaders.push(super::shader(stage, &container));
    }
    let pending = Pending::new(shaders, graphics);
    anyhow::ensure!(
        path.file_name().and_then(|name| name.to_str()) == Some(pending.key()),
        "recipe does not match its name"
    );
    Ok(Arc::new(pending))
}

#[derive(Default, Debug)]
pub struct Report {
    pub built: usize,
    pub refused: usize,
    pub unreadable: usize,
    pub cached: usize,
    pub remaining: usize,
}

fn completed(text: &str) -> HashMap<String, bool> {
    text.lines()
        .filter_map(|line| {
            let (status, key) = line.split_at_checked(1)?;
            let built = match status {
                "+" => true,
                "-" => false,
                _ => return None,
            };
            (key.len() == 32 && key.bytes().all(|byte| byte.is_ascii_hexdigit()))
                .then(|| (key.to_owned(), built))
        })
        .collect()
}

fn checkpoint(path: &Path, entries: &mut Vec<(String, bool)>) -> anyhow::Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    anyhow::ensure!(
        super::library::save_checked(true),
        "pipeline cache could not be saved; preparation progress was not checkpointed"
    );
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for (key, built) in entries.iter() {
        writeln!(file, "{}{key}", if *built { '+' } else { '-' })?;
    }
    file.sync_data()?;
    entries.clear();
    Ok(())
}

fn bake_header(reader: &mut impl io::Read) -> io::Result<bool> {
    let mut header = [0; 5];
    reader.read_exact(&mut header)?;
    let count = usize::from(header[4]);
    if &header[..4] != b"VPR1" || count == 0 || count > STAGES.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid recipe header",
        ));
    }
    let mut records = [0; 8 * 33];
    reader.read_exact(&mut records[..count * 33])?;
    let mut found = false;
    for record in records[..count * 33].chunks_exact(33) {
        let Some(stage) = STAGES.get(usize::from(record[0])) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid shader stage",
            ));
        };
        if *stage == "pixel" {
            let hash = blake3::Hash::from_bytes(record[1..].try_into().unwrap());
            found |= crate::draw::BAKE_PIXEL_SHADERS.contains(&&hash.to_hex()[..16]);
        }
    }
    Ok(found)
}

fn bake_recipes(
    dir: &Path,
    wanted: impl Fn(&str) -> bool,
) -> anyhow::Result<(Vec<PathBuf>, usize)> {
    let mut files = Vec::new();
    let mut unreadable = 0;
    for entry in std::fs::read_dir(dir.join("states"))? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(key) = name
            .to_str()
            .filter(|key| key.len() == 32 && key.bytes().all(|byte| byte.is_ascii_hexdigit()))
        else {
            continue;
        };
        if !wanted(key) || !entry.file_type()?.is_file() {
            continue;
        }
        let file = entry.path();
        let selected = std::fs::File::open(&file).and_then(|mut reader| bake_header(&mut reader));
        match selected {
            Ok(true) => files.push(file),
            Ok(false) => {}
            Err(error) => {
                unreadable += 1;
                tracing::warn!(%error, key, "requested recipe header could not be read for bake preparation");
            }
        }
    }
    files.sort_unstable();
    Ok((files, unreadable))
}

pub(super) fn precompile_bakes(jobs: usize) -> anyhow::Result<Report> {
    anyhow::ensure!(
        PREPARED.get().is_none(),
        "pipeline preparation has already run in this process"
    );
    let dir = DIR
        .get()
        .ok_or_else(|| anyhow::anyhow!("pipeline recipe cache is not configured"))?;
    let (files, unreadable) = bake_recipes(dir, super::library::wanted_early)?;
    let mut report = Report {
        unreadable,
        ..Report::default()
    };
    let mut ready = HashMap::new();
    println!("bake pipeline recipes {}", files.len());
    for batch in files.chunks(128) {
        let remaining = Mutex::new(batch.iter());
        let results = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..jobs.clamp(1, 16).min(batch.len()) {
                scope.spawn(|| {
                    loop {
                        let Some(file) = remaining.lock().unwrap_or_else(|e| e.into_inner()).next()
                        else {
                            break;
                        };
                        let result = load(dir, file).map(|pipeline| {
                            let built = make::now(&pipeline);
                            (pipeline, built)
                        });
                        results
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push((file, result));
                    }
                });
            }
        });
        for (file, result) in results.into_inner().unwrap_or_else(|e| e.into_inner()) {
            match result {
                Ok((pipeline, true)) => {
                    report.built += 1;
                    ready.insert(
                        pipeline.key().to_owned(),
                        Prepared {
                            active: Arc::downgrade(&pipeline),
                            waiting: Some(pipeline),
                        },
                    );
                }
                Ok((pipeline, false)) => {
                    report.refused += 1;
                    tracing::error!(
                        key = pipeline.key(),
                        "required bake pipeline could not be prepared"
                    );
                }
                Err(error) => {
                    report.unreadable += 1;
                    tracing::error!(%error, file = %file.display(), "required bake pipeline recipe could not be loaded");
                }
            }
        }
    }
    PREPARED
        .set(Mutex::new(ready))
        .map_err(|_| anyhow::anyhow!("pipeline preparation was initialized concurrently"))?;
    println!(
        "bake pipeline states {} ready, {} refused, {} unreadable",
        report.built, report.refused, report.unreadable
    );
    Ok(report)
}
pub(super) fn precompile(
    jobs: usize,
    used_only: bool,
    retain_used: bool,
) -> anyhow::Result<Report> {
    let dir = DIR
        .get()
        .ok_or_else(|| anyhow::anyhow!("pipeline recipe cache is not configured"))?;
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir.join("states"))? {
        let entry = entry?;
        let name = entry.file_name();
        if name
            .to_str()
            .is_some_and(|name| name.len() == 32 && name.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            let was_used = name.to_str().is_some_and(super::library::wanted_early);
            if !used_only || was_used {
                files.push((!was_used, entry.path()));
            }
        }
    }
    files.sort_unstable();
    let available = files.len();
    match std::env::var("VICTORY_PRECOMPILE_TOTAL") {
        Ok(value) => {
            let limit = value.parse::<std::num::NonZeroUsize>().map_err(|_| {
                anyhow::anyhow!("VICTORY_PRECOMPILE_TOTAL must be a positive integer")
            })?;
            files.truncate(limit.get());
            println!("pipeline selection {} of {available} recipes", files.len());
        }
        Err(std::env::VarError::NotPresent) => {}
        Err(error) => return Err(error.into()),
    }
    let mut report = Report::default();
    let total = files.len();
    println!("pipeline recipes  {total}");
    let mut identity = blake3::Hasher::new();
    identity.update(
        crate::shader_cache::precompile_setup()
            .map_err(anyhow::Error::msg)?
            .as_bytes(),
    );
    identity.update(include_bytes!("host.rs"));
    identity.update(include_bytes!("../root.rs"));
    identity.update(b"pipeline-preparation-journal-v1");
    let journal = dir.join(format!("prepared-{}.txt", identity.finalize().to_hex()));
    if !super::library::warm_cache_loaded() && journal.exists() {
        std::fs::remove_file(&journal)?;
    }
    let previous = match std::fs::read_to_string(&journal) {
        Ok(text) => completed(&text),
        Err(error) if error.kind() == io::ErrorKind::NotFound => HashMap::new(),
        Err(error) => return Err(error.into()),
    };
    files.retain(|(unused, path)| {
        let Some(key) = path.file_name().and_then(|name| name.to_str()) else {
            return true;
        };
        let Some(&built) = previous.get(key) else {
            return true;
        };
        if retain_used && !unused && built {
            return true;
        }
        report.cached += 1;
        if built {
            report.built += 1
        } else {
            report.refused += 1
        }
        false
    });
    let limit = std::env::var("VICTORY_PRECOMPILE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
        .max(1);
    report.remaining = files.len().saturating_sub(limit);
    files.truncate(limit);
    println!(
        "pipeline prepared {} from cache, {} to process in this worker, {} remaining",
        report.cached,
        files.len(),
        report.remaining
    );
    let mut ready = HashMap::new();
    let mut updated = Vec::new();
    for (index, batch) in files.chunks(128).enumerate() {
        let prepared = Mutex::new(Vec::new());
        let remaining = Mutex::new(batch.iter());
        std::thread::scope(|scope| {
            for _ in 0..jobs.min(batch.len()) {
                scope.spawn(|| {
                    loop {
                        let Some((unused, file)) =
                            remaining.lock().unwrap_or_else(|e| e.into_inner()).next()
                        else {
                            break;
                        };
                        let result = load(dir, file).map(|pipeline| {
                            let built = make::now(&pipeline);
                            (pipeline, built, !unused)
                        });
                        prepared
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(result);
                    }
                });
            }
        });
        for result in prepared.into_inner().unwrap_or_else(|e| e.into_inner()) {
            match result {
                Ok((pipeline, built, used)) => {
                    updated.push((pipeline.key().to_owned(), built));
                    if built {
                        report.built += 1;
                    } else {
                        report.refused += 1;
                    }
                    if used && retain_used {
                        ready.insert(
                            pipeline.key().to_owned(),
                            Prepared {
                                active: Arc::downgrade(&pipeline),
                                waiting: Some(pipeline),
                            },
                        );
                    }
                }
                Err(error) => {
                    report.unreadable += 1;
                    tracing::warn!(%error, "pipeline recipe could not be loaded");
                }
            }
        }
        if (index + 1).is_multiple_of(4) {
            checkpoint(&journal, &mut updated)?;
        }
        println!(
            "pipeline states   {}/{total} processed",
            report.cached + index * 128 + batch.len()
        );
    }
    checkpoint(&journal, &mut updated)?;
    let _ = PREPARED.set(Mutex::new(ready));
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_journal_ignores_torn_records_and_keeps_latest_outcome() {
        let first = "0123456789abcdef0123456789abcdef";
        let second = "abcdef0123456789abcdef0123456789";
        let text = format!(
            "+{first}\n-{second}\n-{first}\n+{first}\n+bad\n?{second}\n+{second}truncated\n"
        );
        let entries = completed(&text);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries.get(first), Some(&true));
        assert_eq!(entries.get(second), Some(&false));
        assert!(completed("+0123456789abcdef").is_empty());
    }

    #[test]
    fn pipeline_recipes_reload_states_and_reject_changed_shader_bytes() {
        let folder = tempfile::tempdir().unwrap();
        let dir = folder.path();
        std::fs::create_dir(dir.join("shaders")).unwrap();
        std::fs::create_dir(dir.join("states")).unwrap();
        let shaders = vec![
            super::super::shader("vertex", b"recipe vertex"),
            super::super::shader("pixel", b"recipe pixel"),
        ];
        let mut graphics = GraphicsState::default();
        graphics.targets.NumRenderTargets = 1;
        graphics.targets.RTFormats[0] =
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
        let original = Pending::new(shaders, Some(Box::new(graphics)));
        save(dir, &original).unwrap();
        save(dir, &original).unwrap();
        assert_eq!(std::fs::read_dir(dir.join("states")).unwrap().count(), 1);
        let path = dir.join("states").join(original.key());
        let restored = load(dir, &path).unwrap();
        assert_eq!(restored.key(), original.key());
        assert_eq!(restored.shaders()[0].container, b"recipe vertex");
        assert_eq!(
            restored.graphics.as_ref().unwrap().describe(),
            original.graphics.as_ref().unwrap().describe()
        );
        let hash = blake3::hash(b"recipe pixel");
        std::fs::write(dir.join("shaders").join(hash.to_hex().as_str()), b"changed").unwrap();
        assert!(load(dir, &path).is_err());
    }
    fn stage_record(stage: u8, prefix: &str) -> [u8; 33] {
        let mut record = [0x9a; 33];
        record[0] = stage;
        for i in 0..8 {
            record[i + 1] = u8::from_str_radix(&prefix[i * 2..i * 2 + 2], 16).unwrap();
        }
        record
    }

    fn recipe_header(records: &[[u8; 33]]) -> Vec<u8> {
        let mut bytes = b"VPR1".to_vec();
        bytes.push(records.len() as u8);
        for record in records {
            bytes.extend(record);
        }
        bytes
    }

    #[test]
    fn bake_selection_recognizes_all_known_pixel_hashes_only_at_pixel_stage() {
        for hash in crate::draw::BAKE_PIXEL_SHADERS {
            let bytes =
                recipe_header(&[stage_record(1, "0123456789abcdef"), stage_record(2, hash)]);
            assert!(bake_header(&mut bytes.as_slice()).unwrap());
            for stage in [0, 1, 3, 4, 5, 6, 7] {
                let bytes = recipe_header(&[stage_record(stage, hash)]);
                assert!(!bake_header(&mut bytes.as_slice()).unwrap());
            }
        }
        let bytes = recipe_header(&[stage_record(2, "0123456789abcdef")]);
        assert!(!bake_header(&mut bytes.as_slice()).unwrap());
    }

    #[test]
    fn bake_selection_reads_only_the_bounded_recipe_header() {
        let mut records = vec![stage_record(1, "0123456789abcdef"); 8];
        records[7] = stage_record(2, crate::draw::BAKE_PIXEL_SHADERS[0]);
        let mut bytes = recipe_header(&records);
        bytes.extend([0xff; 2048]);
        let mut reader = io::Cursor::new(bytes);
        assert!(bake_header(&mut reader).unwrap());
        assert_eq!(reader.position(), 5 + 8 * 33);
    }

    #[test]
    fn bake_selection_reports_corrupt_headers_without_accepting_partial_records() {
        for bytes in [
            b"".to_vec(),
            b"VPR1".to_vec(),
            b"VPR1\0".to_vec(),
            b"VPR1\x09".to_vec(),
            b"VPR2\x01".to_vec(),
            b"VPR1\x01\x02".to_vec(),
            recipe_header(&[stage_record(8, crate::draw::BAKE_PIXEL_SHADERS[0])]),
            recipe_header(&[
                stage_record(2, crate::draw::BAKE_PIXEL_SHADERS[0]),
                stage_record(8, "0123456789abcdef"),
            ]),
        ] {
            assert!(
                bake_header(&mut bytes.as_slice()).is_err(),
                "accepted {bytes:?}"
            );
        }
    }

    #[test]
    fn bake_recipes_filter_names_and_requested_keys_before_reading_any_shader() {
        let folder = tempfile::tempdir().unwrap();
        let dir = folder.path();
        let states = dir.join("states");
        std::fs::create_dir(&states).unwrap();
        let selected = "0123456789abcdef0123456789abcdef";
        let other = "1123456789abcdef0123456789abcdef";
        let unused = "2123456789abcdef0123456789abcdef";
        let bad_name = "3123456789abcdef0123456789abcdef.tmp";
        let header = recipe_header(&[stage_record(2, crate::draw::BAKE_PIXEL_SHADERS[0])]);
        std::fs::write(states.join(selected), &header).unwrap();
        std::fs::write(
            states.join(other),
            recipe_header(&[stage_record(2, "0123456789abcdef")]),
        )
        .unwrap();
        std::fs::write(states.join(unused), b"unreadable if opened").unwrap();
        std::fs::write(states.join(bad_name), b"unreadable if opened").unwrap();
        std::fs::create_dir(states.join("4123456789abcdef0123456789abcdef")).unwrap();
        let (files, unreadable) = bake_recipes(dir, |key| key != unused).unwrap();
        assert_eq!(files, [states.join(selected)]);
        assert_eq!(unreadable, 0);
        assert!(!dir.join("shaders").exists());
    }

    #[test]
    fn bake_recipes_report_requested_broken_headers_and_select_valid_other_recipes() {
        let folder = tempfile::tempdir().unwrap();
        let dir = folder.path();
        let states = dir.join("states");
        std::fs::create_dir(&states).unwrap();
        let good = "abcdef0123456789abcdef0123456789";
        let bad = "0123456789abcdef0123456789abcdef";
        std::fs::write(
            states.join(good),
            recipe_header(&[stage_record(2, crate::draw::BAKE_PIXEL_SHADERS[3])]),
        )
        .unwrap();
        std::fs::write(states.join(bad), b"VPR1\x02").unwrap();
        let (files, unreadable) = bake_recipes(dir, |_| true).unwrap();
        assert_eq!(files, [states.join(good)]);
        assert_eq!(unreadable, 1);
    }

    #[test]
    fn prepared_states_stay_alive_until_runtime_takes_and_reuses_them() {
        let pipeline = Arc::new(Pending::new(Vec::new(), None));
        let weak = Arc::downgrade(&pipeline);
        let mut state = Prepared {
            active: weak.clone(),
            waiting: Some(pipeline),
        };
        assert_eq!(weak.strong_count(), 1);
        let runtime = state.take().unwrap();
        assert!(state.waiting.is_none());
        let second_runtime = state.take().unwrap();
        assert!(Arc::ptr_eq(&runtime, &second_runtime));
        drop(runtime);
        drop(second_runtime);
        assert!(state.take().is_none());
    }
}
