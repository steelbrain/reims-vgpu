//! Generational task/reference namespaces for semantic objects without storage.

use reims_vgpu_protocol::{ResourceId, SerializerRef, TaskId};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NamespaceError {
    IdentitySpaceExhausted,
}

#[derive(Debug)]
struct Slot<M> {
    index: u32,
    next_generation: u32,
    current: Option<ResourceId<M>>,
}

/// One API-specific reference namespace, partitioned by task.
///
/// Samplers, pipelines, heaps, and fences allocate references independently.
/// The marker `M` keeps equal integers in those namespaces non-interchangeable,
/// while the generation makes deletion followed by reference reuse a new
/// internal lifetime.
#[derive(Debug)]
pub struct ReferenceNamespace<M> {
    slots: BTreeMap<(TaskId, SerializerRef<M>), Slot<M>>,
    next_index: u32,
}

impl<M> Default for ReferenceNamespace<M> {
    fn default() -> Self {
        Self {
            slots: BTreeMap::new(),
            next_index: 1,
        }
    }
}

impl<M> ReferenceNamespace<M> {
    pub fn publish(
        &mut self,
        task: TaskId,
        object: SerializerRef<M>,
    ) -> Result<ResourceId<M>, NamespaceError> {
        if let Some(current) = self
            .slots
            .get(&(task, object))
            .and_then(|slot| slot.current)
        {
            return Ok(current);
        }
        let slot = if let Some(slot) = self.slots.get_mut(&(task, object)) {
            slot
        } else {
            let index = self.next_index;
            self.next_index = self
                .next_index
                .checked_add(1)
                .ok_or(NamespaceError::IdentitySpaceExhausted)?;
            self.slots.entry((task, object)).or_insert(Slot {
                index,
                next_generation: 1,
                current: None,
            })
        };
        let id = ResourceId::new(slot.index, slot.next_generation);
        slot.next_generation = slot
            .next_generation
            .checked_add(1)
            .ok_or(NamespaceError::IdentitySpaceExhausted)?;
        slot.current = Some(id);
        Ok(id)
    }

    pub fn resolve(&self, task: TaskId, object: SerializerRef<M>) -> Option<ResourceId<M>> {
        self.slots
            .get(&(task, object))
            .and_then(|slot| slot.current)
    }

    pub fn release(&mut self, task: TaskId, object: SerializerRef<M>) -> bool {
        self.slots
            .get_mut(&(task, object))
            .and_then(|slot| slot.current.take())
            .is_some()
    }

    pub fn release_task(&mut self, task: TaskId) -> usize {
        let mut released = 0;
        for ((slot_task, _), slot) in &mut self.slots {
            if *slot_task == task && slot.current.take().is_some() {
                released += 1;
            }
        }
        released
    }
}

struct TaskReferenceState<T, M> {
    id: ResourceId<M>,
    value: Arc<T>,
}

struct TaskReferenceStateRegistry<T, M> {
    values: BTreeMap<(TaskId, SerializerRef<M>), TaskReferenceState<T, M>>,
    namespace: ReferenceNamespace<M>,
}

impl<T, M> Default for TaskReferenceStateRegistry<T, M> {
    fn default() -> Self {
        Self {
            values: BTreeMap::new(),
            namespace: ReferenceNamespace::default(),
        }
    }
}

/// Retained immutable values in one API-specific task/reference namespace.
///
/// Entries have no capacity or eviction policy. Explicit reference deletion or
/// task teardown are the only retirement events, and reference reuse receives a
/// new generational identity even when the wire integer repeats.
pub struct TaskReferenceStates<T, M>(Mutex<TaskReferenceStateRegistry<T, M>>);

impl<T, M> Default for TaskReferenceStates<T, M> {
    fn default() -> Self {
        Self(Mutex::new(TaskReferenceStateRegistry::default()))
    }
}

impl<T, M> core::fmt::Debug for TaskReferenceStates<T, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let states = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f.debug_struct("TaskReferenceStates")
            .field("entries", &states.values.len())
            .finish()
    }
}

impl<T, M> TaskReferenceStates<T, M> {
    pub fn contains(&self, task_id: u32, reference: SerializerRef<M>) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values
            .contains_key(&(TaskId::new(task_id), reference))
    }

    pub fn get(&self, task_id: u32, reference: SerializerRef<M>) -> Option<Arc<T>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values
            .get(&(TaskId::new(task_id), reference))
            .map(|state| Arc::clone(&state.value))
    }

    pub fn register(&self, task_id: u32, reference: SerializerRef<M>, value: Arc<T>) -> Arc<T> {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = registry.values.get(&(TaskId::new(task_id), reference)) {
            return Arc::clone(&existing.value);
        }
        let id = registry
            .namespace
            .publish(TaskId::new(task_id), reference)
            .expect("semantic reference identity space remains available");
        registry.values.insert(
            (TaskId::new(task_id), reference),
            TaskReferenceState {
                id,
                value: Arc::clone(&value),
            },
        );
        value
    }

    pub fn identity(&self, task_id: u32, reference: SerializerRef<M>) -> Option<ResourceId<M>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values
            .get(&(TaskId::new(task_id), reference))
            .map(|state| state.id)
    }

    pub fn delete(&self, task_id: u32, reference: SerializerRef<M>) -> bool {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed = registry
            .values
            .remove(&(TaskId::new(task_id), reference))
            .is_some();
        if removed {
            assert!(registry.namespace.release(TaskId::new(task_id), reference));
        }
        removed
    }

    pub fn delete_task(&self, task_id: u32) -> usize {
        let mut states = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = states.values.len();
        states
            .values
            .retain(|&(task, _), _| task != TaskId::new(task_id));
        let removed = before - states.values.len();
        assert_eq!(removed, states.namespace.release_task(TaskId::new(task_id)));
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum Sampler {}

    #[test]
    fn reference_reuse_advances_generation_and_tasks_are_independent() {
        let mut namespace = ReferenceNamespace::<Sampler>::default();
        let object = SerializerRef::new(7);
        let first = namespace.publish(TaskId::new(1), object).unwrap();
        assert_eq!(namespace.publish(TaskId::new(1), object).unwrap(), first);
        let other_task = namespace.publish(TaskId::new(2), object).unwrap();
        assert_ne!(first, other_task);

        assert!(namespace.release(TaskId::new(1), object));
        let replacement = namespace.publish(TaskId::new(1), object).unwrap();
        assert_eq!(first.index(), replacement.index());
        assert_ne!(first.generation(), replacement.generation());
        assert_eq!(namespace.release_task(TaskId::new(2)), 1);
        assert_eq!(namespace.resolve(TaskId::new(2), object), None);
    }

    #[test]
    fn retained_values_follow_reference_and_task_lifetimes() {
        let states = TaskReferenceStates::<String, Sampler>::default();
        let reference = SerializerRef::new(4);
        let first = states.register(1, reference, Arc::new("first".to_owned()));
        let first_id = states.identity(1, reference).unwrap();
        let ignored = states.register(1, reference, Arc::new("ignored".to_owned()));
        assert!(Arc::ptr_eq(&first, &ignored));

        assert!(states.delete(1, reference));
        let replacement = states.register(1, reference, Arc::new("replacement".to_owned()));
        assert_eq!(replacement.as_str(), "replacement");
        assert_ne!(states.identity(1, reference), Some(first_id));
        assert_eq!(states.delete_task(1), 1);
        assert!(!states.contains(1, reference));
    }
}
