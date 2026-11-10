//! Sparse task address reservations. Only guest-mapped pages consume section views.
//!
//! A GPU task reserves far more virtual address space than it uses. Splitting the initial
//! placeholder only at mapped pages keeps setup/teardown proportional to those mappings,
//! rather than committing the whole task or allocating an entry for every absent 4 KiB page.
use super::mapping::{PAGE, Page, Section};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::io;
use std::sync::Arc;

#[link(name = "onecore")]
unsafe extern "system" {
    fn VirtualAlloc2(
        process: *mut c_void,
        base: *mut c_void,
        size: usize,
        kind: u32,
        protect: u32,
        parameters: *const c_void,
        count: u32,
    ) -> *mut c_void;
}
#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
}

struct Span {
    page: Page,
    maximum: i32,
}
pub(super) struct Reservation {
    base: usize,
    length: usize,
    spans: BTreeMap<usize, Span>,
}

impl Reservation {
    /// Reserve a no-access address range. No pagefile section or committed memory is created.
    pub(super) fn new(length: usize) -> io::Result<Self> {
        if length == 0 || !length.is_multiple_of(PAGE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "task extent must be whole pages",
            ));
        }
        // SAFETY: a new placeholder reservation, with its size checked above.
        let base = unsafe {
            VirtualAlloc2(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                length,
                0x42000,
                1,
                std::ptr::null(),
                0,
            )
        } as usize;
        if base == 0 {
            return Err(io::Error::last_os_error());
        }
        let span = Span {
            page: Page {
                address: base,
                length,
                backing: None,
                protection: 1,
            },
            maximum: 0,
        };
        Ok(Self {
            base,
            length,
            spans: BTreeMap::from([(0, span)]),
        })
    }
    pub(super) fn address(&self) -> usize {
        self.base
    }
    pub(super) fn length(&self) -> usize {
        self.length
    }
    pub(super) fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
    fn span_at(&self, offset: usize) -> Option<(usize, &Span)> {
        let (&start, span) = self.spans.range(..=offset).next_back()?;
        (offset < start + span.page.length).then_some((start, span))
    }
    pub(super) fn contains(&self, offset: usize) -> bool {
        self.span_at(offset).is_some()
    }
    pub(super) fn maximum(&self, offset: usize) -> Option<i32> {
        self.span_at(offset).map(|(_, s)| s.maximum)
    }
    pub(super) fn backing(&self, offset: usize) -> Option<(Arc<Section>, usize, i32)> {
        let (start, span) = self.span_at(offset)?;
        let (section, file_offset) = span.page.backing.as_ref()?;
        Some((
            Arc::clone(section),
            file_offset + offset - start,
            span.maximum,
        ))
    }
    /// Split an untouched placeholder at a whole-page boundary. Mapped spans are one page,
    /// so this never unmaps or temporarily changes the neighbouring bytes of a live GPU import.
    fn split(&mut self, offset: usize) -> io::Result<()> {
        let Some((start, span)) = self.span_at(offset) else {
            return Ok(());
        };
        if start == offset {
            return Ok(());
        }
        assert!(
            span.page.backing.is_none(),
            "attempted to split a live GPU page"
        );
        let head = offset - start;
        let tail = span.page.length - head;
        // SAFETY: preserving both resulting reservations; only this placeholder is split.
        if unsafe { VirtualFree(span.page.address as *mut _, head, 0x8002) } == 0 {
            return Err(io::Error::last_os_error());
        }
        self.spans
            .get_mut(&start)
            .unwrap_or_else(|| super::failure::abort(concat!(module_path!(), ":", line!())))
            .page
            .length = head;
        self.spans.insert(
            offset,
            Span {
                page: Page {
                    address: self.base + offset,
                    length: tail,
                    backing: None,
                    protection: 1,
                },
                maximum: 0,
            },
        );
        Ok(())
    }
    /// Replace one owned page. A refused kernel map restores the previous mapping or placeholder.
    pub(super) fn replace(
        &mut self,
        offset: usize,
        backing: Option<(Arc<Section>, usize, i32)>,
    ) -> io::Result<()> {
        if !offset.is_multiple_of(PAGE) || !self.contains(offset) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unreserved task page",
            ));
        }
        self.split(offset)?;
        self.split(offset + PAGE)?;
        let span = self
            .spans
            .get_mut(&offset)
            .unwrap_or_else(|| super::failure::abort(concat!(module_path!(), ":", line!())));
        let previous = span.page.backing.clone();
        let old_protection = span.page.protection;
        span.page.unmap()?;
        if let Some((section, file_offset, maximum)) = backing {
            let protection = match maximum {
                0 => 1,
                1 => 2,
                2 | 3 => 4,
                _ => unreachable!("validated VM protection"),
            };
            if let Err(error) = span.page.map(section, file_offset, protection) {
                if let Some((section, file_offset)) = previous {
                    span.page
                        .map(section, file_offset, old_protection)
                        .unwrap_or_else(|error| {
                            super::failure::failed(error, "restoring task page failed")
                        });
                }
                return Err(error);
            }
            span.maximum = maximum;
        } else {
            span.maximum = 0;
            span.page.protection = 1;
        }
        Ok(())
    }
    pub(super) fn protect(&mut self, offset: usize, maximum: i32) -> io::Result<()> {
        let (_, span) = self
            .span_at(offset)
            .unwrap_or_else(|| super::failure::abort("validated task page"));
        if span.page.backing.is_none() {
            assert_eq!(maximum, 0, "an unmapped task page cannot acquire access");
            return Ok(());
        }
        let span = self
            .spans
            .get_mut(&offset)
            .unwrap_or_else(|| super::failure::abort("mapped task pages have whole-page entries"));
        let protection = match maximum {
            0 => 1,
            1 => 2,
            2 | 3 => 4,
            _ => unreachable!("validated VM protection"),
        };
        span.page.protect(protection)?;
        span.maximum = maximum;
        Ok(())
    }
    /// Release intersecting spans, including large untouched placeholders, without a page walk.
    pub(super) fn release(&mut self, start: usize, end: usize) -> io::Result<()> {
        self.split(start)?;
        self.split(end)?;
        let keys: Vec<_> = self.spans.range(start..end).map(|(&at, _)| at).collect();
        for at in keys {
            self.spans.remove(&at);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn huge_task_maps_only_its_first_and_last_pages() {
        let length = 64usize * 1024 * 1024 * 1024;
        let mut task = Reservation::new(length).unwrap();
        assert_eq!(task.spans.len(), 1);
        let section = Section::new(PAGE).unwrap();
        task.replace(0, Some((Arc::clone(&section), 0, 3))).unwrap();
        task.replace(length - PAGE, Some((section, 0, 1))).unwrap();
        assert_eq!(task.spans.len(), 3);
        // SAFETY: the first page is writable and the last is a readable alias of its section.
        unsafe {
            (task.address() as *mut u8).write_volatile(0xd5);
            assert_eq!(
                ((task.address() + length - PAGE) as *const u8).read_volatile(),
                0xd5
            );
        }
        task.release(PAGE, length).unwrap();
        assert_eq!(task.spans.len(), 1);
        assert!(!task.contains(length - PAGE));
    }
}
