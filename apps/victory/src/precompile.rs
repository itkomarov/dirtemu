#[cfg(windows)]
mod windows {
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::{Context, Result};

    use crate::{Console, shader};

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        cfg: &victory_core::Config,
        exe: &Path,
        title_hash: &str,
        console: Console,
        inputs: Option<&[PathBuf]>,
        jobs: Option<usize>,
        used_only: bool,
        retain_used: bool,
        shaders_only: bool,
    ) -> Result<()> {
        let jobs = jobs
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(2, usize::from))
            .clamp(1, 16);
        tracing::info!(jobs, shaders_only, "preparation threads");
        let tag = victory_gpu::shader_cache::precompile_setup().map_err(anyhow::Error::msg)?;
        let worker = !shaders_only
            && std::env::var("VICTORY_PRECOMPILE_WORKER").is_ok_and(|value| value == "1");
        if !worker {
            let cache = cfg.cache_subdir("shader-scan")?;
            let paths = match inputs {
                Some(paths) if !paths.is_empty() => paths.to_vec(),
                _ => files(exe)?,
            };
            let seen = Mutex::new(HashSet::new());
            for path in paths {
                scan(&path, &cache, tag, jobs, &seen)?;
            }
        }
        if shaders_only {
            println!(
                "shader preparation complete; pipeline prebuild skipped ({}, {})",
                console_name(console),
                &title_hash[..12]
            );
            return Ok(());
        }
        let report = if worker {
            victory_gpu::pipeline::precompile_pipelines(jobs, used_only, false)?
        } else {
            let report = prepare_isolated(exe, console, jobs, used_only)?;
            if retain_used {
                println!(
                    "pipeline objects  loading previously used states from the prepared cache"
                );
                victory_gpu::pipeline::precompile_pipelines(jobs, true, true)?;
            }
            report
        };
        println!(
            "pipeline cache    {} ready, {} refused, {} unreadable ({}, {})",
            report.built,
            report.refused,
            report.unreadable,
            console_name(console),
            &title_hash[..12]
        );
        println!(
            "pipeline result {} {} {} {} {}",
            report.built, report.refused, report.unreadable, report.cached, report.remaining
        );
        Ok(())
    }

    fn parse_report(line: &str) -> Option<victory_gpu::pipeline::PipelinePrecompileReport> {
        let numbers: Vec<usize> = line
            .strip_prefix("pipeline result ")?
            .split_whitespace()
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        let [built, refused, unreadable, cached, remaining] = numbers.as_slice() else {
            return None;
        };
        Some(victory_gpu::pipeline::PipelinePrecompileReport {
            built: *built,
            refused: *refused,
            unreadable: *unreadable,
            cached: *cached,
            remaining: *remaining,
        })
    }

    fn prepare_isolated(
        exe: &Path,
        console: Console,
        jobs: usize,
        used_only: bool,
    ) -> Result<victory_gpu::pipeline::PipelinePrecompileReport> {
        use std::io::{BufRead, BufReader};
        use std::os::windows::process::CommandExt;
        use std::process::{Command, Stdio};
        use std::sync::mpsc;
        use std::time::Duration;
        let binary = std::env::current_exe()?;
        let mut previous = usize::MAX;
        loop {
            let mut command = Command::new(&binary);
            command
                .arg("precompile")
                .arg(exe)
                .arg("--console")
                .arg(console_name(console))
                .arg("--jobs")
                .arg(jobs.to_string())
                .env("VICTORY_PRECOMPILE_WORKER", "1")
                .env("VICTORY_PRECOMPILE_LIMIT", "2048")
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .creation_flags(0x0800_0000);
            if used_only {
                command.arg("--used-only");
            }
            let mut child = command
                .spawn()
                .context("starting pipeline preparation worker")?;
            let output = child
                .stdout
                .take()
                .context("pipeline worker has no output pipe")?;
            let (send, receive) = mpsc::channel();
            let reader = std::thread::spawn(move || {
                for line in BufReader::new(output).lines() {
                    if send.send(line).is_err() {
                        break;
                    }
                }
            });
            let mut report = None;
            let mut failure = None;
            loop {
                match receive.recv_timeout(Duration::from_secs(300)) {
                    Ok(Ok(line)) => {
                        if let Some(result) = parse_report(&line) {
                            report = Some(result);
                        }
                        println!("{line}");
                    }
                    Ok(Err(error)) => {
                        failure = Some(anyhow::Error::from(error));
                        break;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        failure = Some(anyhow::anyhow!(
                            "pipeline worker produced no progress for 300 seconds; completed batches are kept"
                        ));
                        break;
                    }
                }
            }
            if failure.is_some() {
                let _ = child.kill();
            }
            let status = child.wait()?;
            reader
                .join()
                .map_err(|_| anyhow::anyhow!("pipeline worker output reader failed"))?;
            if let Some(error) = failure {
                return Err(error);
            }
            anyhow::ensure!(
                status.success(),
                "pipeline worker exited with {status}; completed batches are kept"
            );
            let report = report.context("pipeline worker did not report its result")?;
            anyhow::ensure!(
                report.unreadable == 0,
                "pipeline preparation found {} unreadable recipes",
                report.unreadable
            );
            if report.remaining == 0 {
                return Ok(report);
            }
            anyhow::ensure!(
                report.remaining < previous,
                "pipeline preparation made no checkpointed progress"
            );
            previous = report.remaining;
        }
    }

    pub fn record(cfg: &victory_core::Config, build: &str, console: Console) -> Result<()> {
        let dir = cfg.cache_subdir(&format!(
            "pipeline-recipes/{build}/{}",
            console_name(console)
        ))?;
        victory_gpu::pipeline::record_pipeline_recipes(dir)?;
        Ok(())
    }

    fn console_name(console: Console) -> &'static str {
        match console {
            Console::SeriesX => "series-x",
            Console::SeriesS => "series-s",
        }
    }

    fn files(exe: &Path) -> Result<Vec<PathBuf>> {
        let folder = exe
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut paths = vec![exe.to_owned()];
        for entry in walkdir::WalkDir::new(folder) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            let ext = entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or_default();
            if name.ends_with(".dll")
                || name.contains("shader") && name.ends_with(".rpf")
                || ["sgd5", "dxbc", "cdxil", "cso"]
                    .iter()
                    .any(|e| ext.eq_ignore_ascii_case(e))
            {
                paths.push(entry.into_path());
            }
        }
        paths.sort_unstable();
        Ok(paths)
    }

    fn scan(
        path: &Path,
        cache: &Path,
        target: blake3::Hash,
        jobs: usize,
        seen: &Mutex<HashSet<blake3::Hash>>,
    ) -> Result<()> {
        let meta =
            std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
        anyhow::ensure!(
            meta.is_file() && meta.len() <= shader::MAX_FILE,
            "unsupported shader input {}",
            path.display()
        );
        let mut key = blake3::Hasher::new();
        key.update(target.as_bytes());
        key.update(std::fs::canonicalize(path)?.as_os_str().as_encoded_bytes());
        key.update(&meta.len().to_le_bytes());
        let modified = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());
        if let Some(time) = modified {
            key.update(&time.as_nanos().to_le_bytes());
        }
        let marker = cache.join(key.finalize().to_hex().as_str());
        if modified.is_some()
            && let Ok(result) = std::fs::read_to_string(&marker)
        {
            println!("shader cache      {}  {}", path.display(), result.trim());
            return Ok(());
        }
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let sources = shader::sources(&bytes);
        let total = sources.len();
        if total == 0 {
            return Ok(());
        }
        println!("shader archive    {}  {total} records", path.display());
        let next = AtomicUsize::new(0);
        let processed = AtomicUsize::new(0);
        let ready = AtomicUsize::new(0);
        let refused = AtomicUsize::new(0);
        let failures = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..jobs.min(total) {
                scope.spawn(|| {
                    while let Some(source) = sources.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if let Some(bytes) = source.bytes()
                            && let Some(size) = bytes.get(24..28)
                                .map(|word| u32::from_le_bytes(word.try_into().unwrap()) as usize)
                            && let Some(container) = bytes.get(..size)
                        {
                            let hash = blake3::hash(container);
                            if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(hash) {
                                match victory_gpu::shader_cache::precompile_shader(container) {
                                    Ok(()) => { ready.fetch_add(1, Ordering::Relaxed); }
                                    Err(error) => {
                                        refused.fetch_add(1, Ordering::Relaxed);
                                        failures.lock().unwrap_or_else(|e| e.into_inner())
                                            .push(format!("{hash}\t{error}"));
                                        tracing::debug!(%error, %hash, "shader precompilation refused");
                                    }
                                }
                            }
                        } else {
                            refused.fetch_add(1, Ordering::Relaxed);
                        }
                        let done = processed.fetch_add(1, Ordering::Relaxed) + 1;
                        if done.is_multiple_of(1024) {
                            println!("shader records    {done}/{total} processed");
                        }
                    }
                });
            }
        });
        let result = format!(
            "{} ready, {} refused, {} records",
            ready.into_inner(),
            refused.into_inner(),
            total
        );
        println!("shader cache      {}  {result}", path.display());
        let mut failures = failures.into_inner().unwrap_or_else(|e| e.into_inner());
        failures.sort_unstable();
        std::fs::write(marker.with_extension("failures.txt"), failures.join("\n"))?;
        if modified.is_some() {
            let temporary = marker.with_extension(format!("{}.tmp", std::process::id()));
            std::fs::write(&temporary, &result)?;
            std::fs::rename(temporary, marker)?;
        }
        Ok(())
    }
    #[cfg(test)]
    mod report_tests {
        use super::parse_report;

        #[test]
        fn worker_results_keep_cached_and_remaining_counts() {
            let report = parse_report("pipeline result 2000 48 0 1024 80000").unwrap();
            assert_eq!(
                (
                    report.built,
                    report.refused,
                    report.unreadable,
                    report.cached,
                    report.remaining
                ),
                (2000, 48, 0, 1024, 80000)
            );
            assert!(parse_report("pipeline result 2000 48 0 1024").is_none());
            assert!(parse_report("pipeline result 2000 bad 0 1024 0").is_none());
            assert!(parse_report("shader records 2000/26503 processed").is_none());
        }
    }
}

#[cfg(windows)]
pub use windows::{prepare, record};
