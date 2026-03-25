//! Decoded Picture Buffer (H.264 spec 8.2.5).
//!
//! Stores decoded reference frames for use by P/B slice motion compensation.
//! Manages short-term reference marking via sliding window (spec 8.2.5.3).

use std::rc::Rc;

use crate::nal::NalUnitType;
use crate::slice::SliceHeader;
use crate::sps::Sps;

/// Reference status of a picture in the DPB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceStatus {
    /// Not used for reference.
    Unused,
    /// Short-term reference (identified by frame_num).
    ShortTerm,
}

/// Immutable decoded picture data shared via Rc.
#[derive(Debug)]
pub struct DecodedPicture {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub frame_num: u32,
    pub pic_order_cnt: i32,
}

/// A DPB entry wrapping an Rc<DecodedPicture> with mutable status.
struct DpbEntry {
    pic: Rc<DecodedPicture>,
    reference: ReferenceStatus,
}

/// The Decoded Picture Buffer.
pub struct Dpb {
    max_ref_frames: usize,
    entries: Vec<DpbEntry>,
    // POC type 0 state
    prev_poc_msb: i32,
    prev_poc_lsb: u32,
}

impl Dpb {
    pub fn new(max_ref_frames: usize) -> Self {
        Self {
            max_ref_frames,
            entries: Vec::new(),
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
        }
    }

    /// Update capacity when a new SPS is parsed.
    pub fn set_max_ref_frames(&mut self, max_ref_frames: u32) {
        self.max_ref_frames = max_ref_frames as usize;
    }

    /// Clear the entire DPB. Called on IDR.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.prev_poc_msb = 0;
        self.prev_poc_lsb = 0;
    }

    /// Insert a decoded picture into the DPB.
    /// Applies sliding window marking if needed (spec 8.2.5.3).
    pub fn insert(&mut self, pic: Rc<DecodedPicture>, reference: ReferenceStatus) {
        if reference == ReferenceStatus::ShortTerm {
            self.sliding_window_mark();
        }
        self.entries.push(DpbEntry { pic, reference });
        self.remove_unused();
    }

    /// Get the list of short-term reference pictures, sorted by descending frame_num.
    /// Used to build ref_pic_list_0 for P slices (spec 8.2.4.2.1).
    pub fn short_term_ref_list(&self) -> Vec<Rc<DecodedPicture>> {
        let mut refs: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.reference == ReferenceStatus::ShortTerm)
            .map(|e| e.pic.clone())
            .collect();
        refs.sort_by(|a, b| b.frame_num.cmp(&a.frame_num));
        refs
    }

    /// Compute Picture Order Count for the current picture (spec 8.2.1).
    pub fn compute_poc(
        &mut self,
        sps: &Sps,
        header: &SliceHeader,
        nal_unit_type: NalUnitType,
        nal_ref_idc: u8,
    ) -> i32 {
        match sps.pic_order_cnt_type {
            0 => self.compute_poc_type0(sps, header, nal_unit_type, nal_ref_idc),
            1 => self.compute_poc_type1(sps, header, nal_unit_type, nal_ref_idc),
            2 => Self::compute_poc_type2(header.frame_num, nal_unit_type, nal_ref_idc),
            _ => 0,
        }
    }

    /// POC type 0 (spec 8.2.1.1) — uses pic_order_cnt_lsb with MSB wrapping.
    fn compute_poc_type0(
        &mut self,
        sps: &Sps,
        header: &SliceHeader,
        nal_unit_type: NalUnitType,
        nal_ref_idc: u8,
    ) -> i32 {
        if nal_unit_type == NalUnitType::SliceIdr {
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
            return 0;
        }

        let max_poc_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let poc_lsb = header.pic_order_cnt_lsb;

        let poc_msb = if poc_lsb < self.prev_poc_lsb
            && (self.prev_poc_lsb - poc_lsb) >= max_poc_lsb / 2
        {
            self.prev_poc_msb + max_poc_lsb as i32
        } else if poc_lsb > self.prev_poc_lsb
            && (poc_lsb - self.prev_poc_lsb) > max_poc_lsb / 2
        {
            self.prev_poc_msb - max_poc_lsb as i32
        } else {
            self.prev_poc_msb
        };

        let poc = poc_msb + poc_lsb as i32;

        if nal_ref_idc > 0 {
            self.prev_poc_msb = poc_msb;
            self.prev_poc_lsb = poc_lsb;
        }

        poc
    }

    /// POC type 1 (spec 8.2.1.2) — uses delta_pic_order_cnt with cycle offsets.
    fn compute_poc_type1(
        &self,
        sps: &Sps,
        header: &SliceHeader,
        nal_unit_type: NalUnitType,
        nal_ref_idc: u8,
    ) -> i32 {
        if nal_unit_type == NalUnitType::SliceIdr {
            return 0;
        }

        let num_ref_frames_in_cycle = sps.num_ref_frames_in_pic_order_cnt_cycle as usize;
        let expected_delta_per_cycle: i32 = sps.offset_for_ref_frame.iter().sum();

        let abs_frame_num = if num_ref_frames_in_cycle > 0 {
            header.frame_num as i32
        } else {
            0
        };

        let expected_poc = if nal_ref_idc == 0 && abs_frame_num > 0 {
            let cycle = (abs_frame_num - 1) / num_ref_frames_in_cycle.max(1) as i32;
            let idx = ((abs_frame_num - 1) % num_ref_frames_in_cycle.max(1) as i32) as usize;
            let partial: i32 = sps.offset_for_ref_frame[..=idx].iter().sum();
            cycle * expected_delta_per_cycle + partial + sps.offset_for_non_ref_pic
        } else if abs_frame_num > 0 {
            let cycle = (abs_frame_num - 1) / num_ref_frames_in_cycle.max(1) as i32;
            let idx = ((abs_frame_num - 1) % num_ref_frames_in_cycle.max(1) as i32) as usize;
            let partial: i32 = sps.offset_for_ref_frame[..=idx].iter().sum();
            cycle * expected_delta_per_cycle + partial
        } else {
            0
        };

        expected_poc + header.delta_pic_order_cnt[0]
    }

    /// POC type 2 (spec 8.2.1.3) — derived directly from frame_num.
    fn compute_poc_type2(frame_num: u32, nal_unit_type: NalUnitType, nal_ref_idc: u8) -> i32 {
        if nal_unit_type == NalUnitType::SliceIdr {
            0
        } else if nal_ref_idc > 0 {
            2 * frame_num as i32
        } else {
            2 * frame_num as i32 - 1
        }
    }

    /// Sliding window reference marking (spec 8.2.5.3).
    /// If short-term ref count >= max_ref_frames, mark the oldest as unused.
    fn sliding_window_mark(&mut self) {
        let short_term_count = self
            .entries
            .iter()
            .filter(|e| e.reference == ReferenceStatus::ShortTerm)
            .count();

        if short_term_count >= self.max_ref_frames && self.max_ref_frames > 0 {
            // Find the short-term ref with the smallest frame_num
            if let Some(idx) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.reference == ReferenceStatus::ShortTerm)
                .min_by_key(|(_, e)| e.pic.frame_num)
                .map(|(i, _)| i)
            {
                self.entries[idx].reference = ReferenceStatus::Unused;
            }
        }
    }

    /// Remove entries that are unused for reference (freeing memory).
    fn remove_unused(&mut self) {
        self.entries
            .retain(|e| e.reference != ReferenceStatus::Unused);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pic(frame_num: u32, poc: i32) -> Rc<DecodedPicture> {
        Rc::new(DecodedPicture {
            y: vec![],
            u: vec![],
            v: vec![],
            width: 16,
            height: 16,
            frame_num,
            pic_order_cnt: poc,
        })
    }

    #[test]
    fn test_dpb_insert_and_clear() {
        let mut dpb = Dpb::new(4);
        dpb.insert(make_pic(0, 0), ReferenceStatus::ShortTerm);
        dpb.insert(make_pic(1, 2), ReferenceStatus::ShortTerm);
        assert_eq!(dpb.short_term_ref_list().len(), 2);
        dpb.clear();
        assert_eq!(dpb.short_term_ref_list().len(), 0);
    }

    #[test]
    fn test_sliding_window() {
        let mut dpb = Dpb::new(2);
        dpb.insert(make_pic(0, 0), ReferenceStatus::ShortTerm);
        dpb.insert(make_pic(1, 2), ReferenceStatus::ShortTerm);
        assert_eq!(dpb.short_term_ref_list().len(), 2);

        // Third insert triggers sliding window — oldest (frame_num=0) gets evicted
        dpb.insert(make_pic(2, 4), ReferenceStatus::ShortTerm);
        let refs = dpb.short_term_ref_list();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].frame_num, 2); // newest first (descending)
        assert_eq!(refs[1].frame_num, 1);
    }

    #[test]
    fn test_poc_type2() {
        assert_eq!(Dpb::compute_poc_type2(0, NalUnitType::SliceIdr, 3), 0);
        assert_eq!(Dpb::compute_poc_type2(1, NalUnitType::Slice, 3), 2);
        assert_eq!(Dpb::compute_poc_type2(1, NalUnitType::Slice, 0), 1);
        assert_eq!(Dpb::compute_poc_type2(5, NalUnitType::Slice, 1), 10);
    }

    #[test]
    fn test_ref_list_sorted_descending() {
        let mut dpb = Dpb::new(4);
        dpb.insert(make_pic(3, 6), ReferenceStatus::ShortTerm);
        dpb.insert(make_pic(1, 2), ReferenceStatus::ShortTerm);
        dpb.insert(make_pic(5, 10), ReferenceStatus::ShortTerm);

        let refs = dpb.short_term_ref_list();
        assert_eq!(refs[0].frame_num, 5);
        assert_eq!(refs[1].frame_num, 3);
        assert_eq!(refs[2].frame_num, 1);
    }

    #[test]
    fn test_unused_not_in_ref_list() {
        let mut dpb = Dpb::new(4);
        dpb.insert(make_pic(0, 0), ReferenceStatus::ShortTerm);
        dpb.insert(make_pic(1, 2), ReferenceStatus::Unused); // non-reference
        assert_eq!(dpb.short_term_ref_list().len(), 1);
    }
}
