//! The display pipe's colour: what the guest's display does to the pixels it shows, as the
//! display pipe's transactions set it (`DISPLAY_COLOR`, `reims_vgpu_graphics::wire`'s
//! [`RgfxDisplayColor`]). Four stages apply in order, after the surface's own colour: plane 0's
//! gamma table, the pipe's gamma table before its colour matrix, the colour matrix, and the gamma
//! table after it. Each is off (it passes its input through) until a transaction sets it, and keeps
//! its table until one sets it again, as a display engine keeps its LUTs across flips. The
//! device's reset turns every stage off, though it keeps the scanout (`Presented::reset` says
//! why).
//!
//! **Staging.** The kext posts a transaction's stages as `DISPLAY_COLOR` parts on the presents'
//! lane, then the transaction's present. [`ColorStaging`] puts each stage's parts together, in
//! order, and holds the stages it has whole for the next present the host shows
//! ([`ColorStaging::take`]), which applies them to the display's colour as the display shows it
//! ([`DisplayColor::apply`]): the colour changes with the frame of the transaction that set it, as
//! a display engine latches its LUTs with the flip. A part out of order, or past the stage's
//! entries, drops what was staged of its stage, and the stage keeps its table; a present that
//! comes while a stage's parts are not all in drops them too.
//!
//! **The identity.** A matrix stage a transaction sets to the identity (rows `[1, 0, 0, 0]`,
//! `[0, 1, 0, 0]` and `[0, 0, 1, 0]`, compared as floats) is staged off: it gives each finite
//! input back exactly (`1 * x + 0 * y + 0 * z + 0` is `x`), so the display shows the red, green
//! and blue it shows without it, and the window draws no colour pass for it (WindowServer sets
//! identity matrices during its fades, and no gamma table: measured on an x86-64 macOS 26 guest).
//! A gamma table that is a linear ramp stays on: its interpolation gives the identity only to
//! within a few ulps, and it clamps its input, so leaving it out could change a pixel of a float
//! surface.
//!
//! **What the window does with it.** The window's sink draws the display through the colour
//! (`sink`) while any stage is on: each gamma stage interpolates linearly between its entries,
//! entry `k` of `n` at input `k / (n - 1)`, its input clamped to [0, 1]; the matrix gives output
//! channel `i` as `m[i][0] * red + m[i][1] * green + m[i][2] * blue + m[i][3]`, the transaction's
//! twelve floats as it carries them, three rows (read from macOS 26's IOPresentment, which puts
//! CoreDisplay's matrix into the transaction's rows, and IOAccelerator, which sends them). How a
//! Mac's display engine samples between entries is not measured.

use std::sync::Arc;

use reims_vgpu_graphics::wire::{
    RGFX_COLOR_MATRIX, RGFX_COLOR_MAX_ENTRIES, RgfxColorEntry, RgfxDisplayColor,
};

/// The stages, by their `RGFX_COLOR_*` number.
pub(crate) const STAGES: usize = 4;

/// A stage's table: its entries, four `f32`s each (a gamma entry's red, green, blue and an unused
/// fourth; a matrix's three rows).
pub(crate) type Table = Arc<[[f32; 4]]>;

/// The display's colour: each stage's table, `None` where it is off.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct DisplayColor {
    /// By `RGFX_COLOR_*` stage.
    pub(crate) stages: [Option<Table>; STAGES],
}

impl DisplayColor {
    /// Whether every stage is off: the display shows its pixels as they are.
    pub(crate) fn is_identity(&self) -> bool {
        self.stages.iter().all(Option::is_none)
    }

    /// The stages `change` sets take their new tables; the others keep theirs.
    pub(crate) fn apply(&mut self, change: ColorChange) {
        for (stage, set) in self.stages.iter_mut().zip(change.stages) {
            if let Some(table) = set {
                *stage = table;
            }
        }
    }
}

/// The stages a present sets: for each, `Some` of its new table (`None` inside: off), or `None`
/// where the stage keeps its table.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ColorChange {
    /// By `RGFX_COLOR_*` stage.
    pub(crate) stages: [Option<Option<Table>>; STAGES],
}

/// The identity matrix's rows (the module doc).
const IDENTITY: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

/// Whether a matrix stage's `rows` are the identity's, compared as floats (`-0.0` is `0.0`).
fn is_identity_matrix(rows: &[[f32; 4]]) -> bool {
    rows.len() == IDENTITY.len() && rows.iter().zip(&IDENTITY).all(|(row, one)| row == one)
}

/// Why a `DISPLAY_COLOR` part was dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dropped {
    /// Its stage is none of the four.
    Stage,
    /// Its stage's entries are past [`RGFX_COLOR_MAX_ENTRIES`], or a matrix's are neither 3 nor 0.
    Entries,
    /// It does not continue its stage's parts: not the next entry, another entry count, or past
    /// the stage's entries.
    Order,
}

/// The `DISPLAY_COLOR` parts staged since the last present (the module doc).
#[derive(Debug, Default)]
pub(crate) struct ColorStaging {
    /// By stage: the stage's entry count, and its entries so far, while its parts come.
    building: [Option<(u32, Vec<[f32; 4]>)>; STAGES],
    /// The stages staged whole.
    staged: ColorChange,
}

impl ColorStaging {
    /// Stages one part, `color`'s fields and the `entries` it carries; a part that completes its
    /// stage stages the stage whole. A part refused drops what was staged of its stage.
    pub(crate) fn part(
        &mut self,
        color: &RgfxDisplayColor,
        entries: &[RgfxColorEntry],
    ) -> Result<(), Dropped> {
        let stage = usize::try_from(color.stage)
            .ok()
            .filter(|&s| s < STAGES)
            .ok_or(Dropped::Stage)?;
        let count = color.entry_count;
        let refused = |building: &mut Option<(u32, Vec<[f32; 4]>)>, why| {
            *building = None;
            Err(why)
        };
        let building = &mut self.building[stage];
        if count > RGFX_COLOR_MAX_ENTRIES
            || (color.stage == RGFX_COLOR_MATRIX && !matches!(count, 0 | 3))
        {
            return refused(building, Dropped::Entries);
        }
        if color.first_entry == 0 {
            *building = Some((count, Vec::new()));
        }
        let Some((want, so_far)) = building.as_mut() else {
            return refused(building, Dropped::Order);
        };
        let first = usize::try_from(color.first_entry).unwrap_or(usize::MAX);
        let room = usize::try_from(count)
            .unwrap_or(0)
            .saturating_sub(so_far.len());
        if *want != count || so_far.len() != first || entries.len() > room {
            return refused(building, Dropped::Order);
        }
        so_far.extend(
            entries
                .iter()
                .map(|e| [e.x, e.y, e.z, e.w].map(f32::from_bits)),
        );
        if so_far.len() == usize::try_from(count).unwrap_or(usize::MAX) {
            let table = std::mem::take(so_far);
            *building = None;
            let on = !table.is_empty()
                && !(color.stage == RGFX_COLOR_MATRIX && is_identity_matrix(&table));
            self.staged.stages[stage] = Some(on.then(|| Table::from(table)));
        }
        Ok(())
    }

    /// The stages staged whole since the last present, for the present the host shows now, or
    /// `None` where none was; a stage whose parts are not all in is dropped.
    pub(crate) fn take(&mut self) -> Option<ColorChange> {
        if self.building.iter().any(Option::is_some) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "a present came before every part of a display colour stage: the stage keeps its \
                 table"
            );
            self.building = Default::default();
        }
        let change = std::mem::take(&mut self.staged);
        change.stages.iter().any(Option::is_some).then_some(change)
    }

    /// The device's reset: nothing staged.
    pub(crate) fn reset(&mut self) {
        *self = ColorStaging::default();
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_graphics::wire::{RGFX_COLOR_POST_GAMMA, RGFX_COLOR_PRE_GAMMA};

    use super::*;

    /// A part of `stage`, entries `first` on of `count`, `part` entries carried.
    fn part(
        stage: u32,
        count: u32,
        first: u32,
        part: u32,
    ) -> (RgfxDisplayColor, Vec<RgfxColorEntry>) {
        let color = RgfxDisplayColor {
            stage,
            entry_count: count,
            first_entry: first,
            part_entries: part,
            ..RgfxDisplayColor::default()
        };
        #[expect(clippy::as_conversions, reason = "small test indices")]
        let entries = (first..first + part)
            .map(|k| {
                let v = (k as f32).to_bits();
                RgfxColorEntry {
                    x: v,
                    y: v,
                    z: v,
                    w: 0,
                }
            })
            .collect();
        (color, entries)
    }

    fn stage(
        staging: &mut ColorStaging,
        (color, entries): (RgfxDisplayColor, Vec<RgfxColorEntry>),
    ) -> Result<(), Dropped> {
        staging.part(&color, &entries)
    }

    /// A stage's parts put together in order make its table, staged for the next present alone;
    /// a stage of no entries is staged off.
    #[test]
    fn parts_in_order_stage_their_table_for_the_next_present() {
        let mut s = ColorStaging::default();
        assert_eq!(s.take(), None);
        stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 3, 0, 2)).unwrap();
        stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 3, 2, 1)).unwrap();
        stage(&mut s, part(RGFX_COLOR_PRE_GAMMA, 0, 0, 0)).unwrap();
        let change = s.take().unwrap();
        let post = change.stages[RGFX_COLOR_POST_GAMMA as usize]
            .clone()
            .unwrap()
            .unwrap();
        assert_eq!(
            &*post,
            &[
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [2.0, 2.0, 2.0, 0.0]
            ]
        );
        assert_eq!(change.stages[RGFX_COLOR_PRE_GAMMA as usize], Some(None));
        assert_eq!(change.stages[RGFX_COLOR_MATRIX as usize], None);
        assert_eq!(s.take(), None);
    }

    /// A part that skips an entry, changes the count, runs past the stage's entries, or comes
    /// without the stage's first, drops what was staged of the stage; a matrix of 2 entries, a
    /// stage past the four and a count past the limit are refused.
    #[test]
    fn parts_out_of_order_drop_their_stage() {
        let mut s = ColorStaging::default();
        stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 4, 0, 2)).unwrap();
        assert_eq!(
            stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 4, 3, 1)),
            Err(Dropped::Order)
        );
        assert_eq!(
            stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 4, 2, 2)),
            Err(Dropped::Order)
        );
        stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 4, 0, 2)).unwrap();
        assert_eq!(
            stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 5, 2, 2)),
            Err(Dropped::Order)
        );
        stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 2, 0, 1)).unwrap();
        assert_eq!(
            stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 2, 1, 2)),
            Err(Dropped::Order)
        );
        assert_eq!(
            stage(&mut s, part(RGFX_COLOR_MATRIX, 2, 0, 2)),
            Err(Dropped::Entries)
        );
        assert_eq!(stage(&mut s, part(4, 1, 0, 1)), Err(Dropped::Stage));
        assert_eq!(
            stage(
                &mut s,
                part(RGFX_COLOR_PRE_GAMMA, RGFX_COLOR_MAX_ENTRIES + 1, 0, 1)
            ),
            Err(Dropped::Entries)
        );
        assert_eq!(s.take(), None);
    }

    /// A present that comes while a stage's parts are not all in drops them: the next present
    /// sets nothing of that stage.
    #[test]
    fn a_present_drops_a_stage_not_all_in() {
        let mut s = ColorStaging::default();
        stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 4, 0, 2)).unwrap();
        stage(&mut s, part(RGFX_COLOR_MATRIX, 3, 0, 3)).unwrap();
        let change = s.take().unwrap();
        assert_eq!(change.stages[RGFX_COLOR_POST_GAMMA as usize], None);
        assert!(change.stages[RGFX_COLOR_MATRIX as usize].is_some());
        assert_eq!(
            stage(&mut s, part(RGFX_COLOR_POST_GAMMA, 4, 2, 2)),
            Err(Dropped::Order)
        );
    }

    /// A matrix stage set to the identity is staged off, `-0.0` counting as `0.0`, and turns off
    /// a matrix set before; a matrix one float off the identity, its offset included, stays on,
    /// and so does a linear gamma ramp
    #[test]
    fn an_identity_matrix_is_staged_off() {
        let staged = |stage: u32, rows: &[[f32; 4]]| {
            let mut s = ColorStaging::default();
            let color = RgfxDisplayColor {
                stage,
                entry_count: u32::try_from(rows.len()).unwrap(),
                part_entries: u32::try_from(rows.len()).unwrap(),
                ..RgfxDisplayColor::default()
            };
            let entries: Vec<RgfxColorEntry> = rows
                .iter()
                .map(|r| {
                    let [x, y, z, w] = r.map(f32::to_bits);
                    RgfxColorEntry { x, y, z, w }
                })
                .collect();
            s.part(&color, &entries).unwrap();
            s.take().unwrap().stages[usize::try_from(stage).unwrap()].clone()
        };
        let mut identity = IDENTITY;
        assert_eq!(staged(RGFX_COLOR_MATRIX, &identity), Some(None));
        identity[0][1] = -0.0;
        assert_eq!(staged(RGFX_COLOR_MATRIX, &identity), Some(None));
        for (row, column, value) in [(0, 0, 0.999_999_9), (1, 2, 1e-30), (2, 3, 1.0 / 255.0)] {
            let mut rows = IDENTITY;
            rows[row][column] = value;
            let on = staged(RGFX_COLOR_MATRIX, &rows);
            assert_eq!(
                on.flatten().as_deref(),
                Some(&rows[..]),
                "[{row}][{column}]"
            );
        }
        let mut nan = IDENTITY;
        nan[1][1] = f32::NAN;
        assert!(staged(RGFX_COLOR_MATRIX, &nan).unwrap().is_some());
        let ramp: Vec<[f32; 4]> = (0..256u16)
            .map(|k| {
                let v = f32::from(k) / 255.0;
                [v, v, v, 0.0]
            })
            .collect();
        for stage in [RGFX_COLOR_PRE_GAMMA, RGFX_COLOR_POST_GAMMA] {
            assert!(staged(stage, &ramp).unwrap().is_some());
        }

        // Applied, the identity turns off the matrix set before.
        let mut color = DisplayColor::default();
        let mut change = ColorChange::default();
        let swap: Table = Arc::from(vec![
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]);
        change.stages[RGFX_COLOR_MATRIX as usize] = Some(Some(swap));
        color.apply(change);
        assert!(!color.is_identity());
        let mut change = ColorChange::default();
        change.stages[RGFX_COLOR_MATRIX as usize] = staged(RGFX_COLOR_MATRIX, &IDENTITY);
        color.apply(change);
        assert!(color.is_identity());
    }

    /// A change sets the stages it names and leaves the others' tables; a colour is the identity
    /// only with every stage off.
    #[test]
    fn a_change_sets_only_its_stages() {
        let mut color = DisplayColor::default();
        assert!(color.is_identity());
        let table: Table = Arc::from(vec![[0.5; 4]; 2]);
        let mut change = ColorChange::default();
        change.stages[RGFX_COLOR_PRE_GAMMA as usize] = Some(Some(Arc::clone(&table)));
        color.apply(change);
        assert!(!color.is_identity());
        let mut change = ColorChange::default();
        change.stages[RGFX_COLOR_POST_GAMMA as usize] = Some(None);
        color.apply(change);
        assert_eq!(color.stages[RGFX_COLOR_PRE_GAMMA as usize], Some(table));
        let mut change = ColorChange::default();
        change.stages[RGFX_COLOR_PRE_GAMMA as usize] = Some(None);
        color.apply(change);
        assert!(color.is_identity());
    }
}
