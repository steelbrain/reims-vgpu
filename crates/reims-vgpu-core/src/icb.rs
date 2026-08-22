//! Task-local semantic indirect-command-buffer registry.

use std::collections::BTreeMap;
use std::sync::Mutex;

use reims_vgpu_protocol::{
    IcbCommandMemory, IndirectCommandBufferDescriptor, IndirectCommandBufferObject, ResourceId,
    SerializerRef, TaskId,
};

use crate::{NamespaceError, ReferenceNamespace};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IcbRecord {
    pub descriptor: IndirectCommandBufferDescriptor,
    pub command_memory: Option<IcbCommandMemory>,
}

#[derive(Debug)]
struct IcbEntry {
    id: ResourceId<IndirectCommandBufferObject>,
    record: IcbRecord,
}

#[derive(Debug, Default)]
struct IcbRegistryInner {
    namespace: ReferenceNamespace<IndirectCommandBufferObject>,
    records: BTreeMap<(TaskId, SerializerRef<IndirectCommandBufferObject>), IcbEntry>,
}

/// Device-owned ICB declarations and their independently bound command memory.
#[derive(Debug, Default)]
pub struct IcbRegistry {
    inner: Mutex<IcbRegistryInner>,
}

impl IcbRegistry {
    fn lock(&self) -> std::sync::MutexGuard<'_, IcbRegistryInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn record(
        &self,
        task_id: u32,
        icb_ref: u32,
        descriptor: IndirectCommandBufferDescriptor,
    ) -> Result<IndirectCommandBufferDescriptor, NamespaceError> {
        let mut inner = self.lock();
        let task = TaskId::new(task_id);
        let object = SerializerRef::new(icb_ref);
        if let Some(entry) = inner.records.get(&(task, object)) {
            if entry.record.descriptor == descriptor {
                return Ok(entry.record.descriptor.clone());
            }
        }
        inner.records.remove(&(task, object));
        inner.namespace.release(task, object);
        let id = inner.namespace.publish(task, object)?;
        inner.records.insert(
            (task, object),
            IcbEntry {
                id,
                record: IcbRecord {
                    descriptor: descriptor.clone(),
                    command_memory: None,
                },
            },
        );
        Ok(descriptor)
    }

    pub fn bind(&self, task_id: u32, icb_ref: u32, memory: IcbCommandMemory) -> bool {
        let mut inner = self.lock();
        let Some(entry) = inner
            .records
            .get_mut(&(TaskId::new(task_id), SerializerRef::new(icb_ref)))
        else {
            return false;
        };
        entry.record.command_memory = Some(memory);
        true
    }

    pub fn snapshot(&self, task_id: u32, icb_ref: u32) -> Option<IcbRecord> {
        let inner = self.lock();
        let task = TaskId::new(task_id);
        let object = SerializerRef::new(icb_ref);
        let entry = inner.records.get(&(task, object))?;
        debug_assert_eq!(inner.namespace.resolve(task, object), Some(entry.id));
        Some(entry.record.clone())
    }

    pub fn delete(&self, task_id: u32, icb_ref: u32) -> bool {
        let mut inner = self.lock();
        let task = TaskId::new(task_id);
        let object = SerializerRef::new(icb_ref);
        let removed = inner.records.remove(&(task, object)).is_some();
        let released = inner.namespace.release(task, object);
        debug_assert_eq!(removed, released);
        removed
    }

    pub fn delete_task(&self, task_id: u32) -> usize {
        let mut inner = self.lock();
        let task = TaskId::new(task_id);
        let before = inner.records.len();
        inner
            .records
            .retain(|&(record_task, _), _| record_task != task);
        let removed = before - inner.records.len();
        debug_assert_eq!(removed, inner.namespace.release_task(task));
        removed
    }

    pub fn identity(
        &self,
        task_id: u32,
        icb_ref: u32,
    ) -> Option<ResourceId<IndirectCommandBufferObject>> {
        self.lock()
            .records
            .get(&(TaskId::new(task_id), SerializerRef::new(icb_ref)))
            .map(|entry| entry.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_reuse_changes_identity_and_drops_old_command_memory() {
        let registry = IcbRegistry::default();
        let descriptor = IndirectCommandBufferDescriptor::default();
        registry.record(1, 9, descriptor.clone()).unwrap();
        let first = registry.identity(1, 9).unwrap();
        assert!(registry.bind(
            1,
            9,
            IcbCommandMemory {
                gva: 0x4000,
                byte_len: 64
            }
        ));
        let mut changed = descriptor;
        changed.max_command_count = 2;
        registry.record(1, 9, changed).unwrap();
        assert_ne!(registry.identity(1, 9), Some(first));
        assert_eq!(registry.snapshot(1, 9).unwrap().command_memory, None);
    }

    #[test]
    fn task_retirement_is_namespace_local() {
        let registry = IcbRegistry::default();
        registry.record(1, 9, Default::default()).unwrap();
        registry.record(2, 9, Default::default()).unwrap();
        assert_eq!(registry.delete_task(1), 1);
        assert!(registry.identity(1, 9).is_none());
        assert!(registry.identity(2, 9).is_some());
    }
}
