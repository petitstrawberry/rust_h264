use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub enum Phase {
    MbDecode,
    MotionComp,
    Deblock,
    DpbFinalize,
    FinalizeMbInfo,
}

#[derive(Clone, Copy)]
pub struct PhaseSample {
    pub name: &'static str,
    pub micros: u64,
}

static MB_DECODE_US: AtomicU64 = AtomicU64::new(0);
static MOTION_COMP_US: AtomicU64 = AtomicU64::new(0);
static DEBLOCK_US: AtomicU64 = AtomicU64::new(0);
static DPB_FINALIZE_US: AtomicU64 = AtomicU64::new(0);
static FINALIZE_MB_INFO_US: AtomicU64 = AtomicU64::new(0);
static ENABLED: AtomicBool = AtomicBool::new(false);

pub struct PhaseTimer {
    phase: Phase,
    start: Option<Instant>,
}

impl PhaseTimer {
    #[inline]
    pub fn start(phase: Phase) -> Self {
        Self {
            phase,
            start: ENABLED.load(Ordering::Relaxed).then(Instant::now),
        }
    }
}

impl Drop for PhaseTimer {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            add_duration(self.phase, start.elapsed());
        }
    }
}

pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn reset() {
    MB_DECODE_US.store(0, Ordering::Relaxed);
    MOTION_COMP_US.store(0, Ordering::Relaxed);
    DEBLOCK_US.store(0, Ordering::Relaxed);
    DPB_FINALIZE_US.store(0, Ordering::Relaxed);
    FINALIZE_MB_INFO_US.store(0, Ordering::Relaxed);
}

pub fn snapshot() -> [PhaseSample; 5] {
    [
        PhaseSample {
            name: "mb_decode_total",
            micros: MB_DECODE_US.load(Ordering::Relaxed),
        },
        PhaseSample {
            name: "motion_comp",
            micros: MOTION_COMP_US.load(Ordering::Relaxed),
        },
        PhaseSample {
            name: "deblock",
            micros: DEBLOCK_US.load(Ordering::Relaxed),
        },
        PhaseSample {
            name: "dpb_finalize",
            micros: DPB_FINALIZE_US.load(Ordering::Relaxed),
        },
        PhaseSample {
            name: "finalize_mb_info",
            micros: FINALIZE_MB_INFO_US.load(Ordering::Relaxed),
        },
    ]
}

#[inline]
fn add_duration(phase: Phase, duration: Duration) {
    let micros = duration.as_micros().min(u128::from(u64::MAX)) as u64;
    match phase {
        Phase::MbDecode => MB_DECODE_US.fetch_add(micros, Ordering::Relaxed),
        Phase::MotionComp => MOTION_COMP_US.fetch_add(micros, Ordering::Relaxed),
        Phase::Deblock => DEBLOCK_US.fetch_add(micros, Ordering::Relaxed),
        Phase::DpbFinalize => DPB_FINALIZE_US.fetch_add(micros, Ordering::Relaxed),
        Phase::FinalizeMbInfo => FINALIZE_MB_INFO_US.fetch_add(micros, Ordering::Relaxed),
    };
}
