//! What a guest's synchronous ring-0 create costs from its post to its answer, as the kext waits
//! for it: a measurement, run by hand (`--run-ignored only --no-capture`).

use std::time::Instant;

use super::*;

/// A host that answers a render pipeline's create at once, with its reflection, as a create the
/// host's caches hold is answered, and carries out every other message doing nothing.
struct AnswerHost;

impl Host for AnswerHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn scanout_written(&self, _offset: u64, _length: u64) -> bool {
        false
    }
    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        if header.r#type == RGFX_CMD_RENDER_PIPELINE_CREATE {
            let mut reflection = RgfxPipelineReflection::new_zeroed();
            reflection.header = RgfxMsgHeader {
                r#type: RGFX_EVT_PIPELINE_REFLECTION,
                size: size_of::<RgfxPipelineReflection>() as u32,
                seqno: header.seqno,
            };
            completion.post(reflection.as_bytes());
        }
    }
}

/// The next event of the completion ring, its seqno, read as soon as it is posted (spinning).
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn answer_now(rig: &mut Rig) -> u64 {
    let data = RGFX_CONTROL_RING_DATA_SIZE as u64;
    let at = |count: u64| RGFX_CONTROL_COMPLETE_OFFSET + 256 + (count % data);
    let start = Instant::now();
    loop {
        let produced = (rig.region())
            .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET)
            .load(Ordering::Acquire);
        if produced != rig.consumed {
            let head = rig.region().read(at(rig.consumed), 16).unwrap();
            let size = u32::from_le_bytes(head[4..8].try_into().unwrap()) as u64;
            let r#type = u32::from_le_bytes(head[0..4].try_into().unwrap());
            let seqno = u64::from_le_bytes(head[8..16].try_into().unwrap());
            rig.consumed += size;
            (rig.region())
                .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET + 0x40)
                .store(rig.consumed, Ordering::Release);
            if r#type != RGFX_MSG_PAD {
                return seqno;
            }
            continue;
        }
        assert!(start.elapsed() < Duration::from_secs(2), "no answer");
        std::hint::spin_loop();
    }
}

/// Microseconds' mean, median and 99th percentile.
fn stats(mut us: Vec<f64>) -> (f64, f64, f64) {
    us.sort_by(f64::total_cmp);
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    (mean, us[us.len() / 2], us[us.len() * 99 / 100])
}

/// A context's synchronous creates, each posted once the one before it is answered, as the kext
/// posts a create it waits for: back to back (`warm`), each after `ASYNC` of the context's creates
/// it does not wait for (`behind`), and each after a pause longer than a lane's thread waits
/// (`cold`). Prints the mean, median and 99th percentile of each, from the post's doorbell to
/// the answer read.
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_sync_create_s_round_trip() {
    const CONTEXT: u64 = 5;
    const ROUNDS: usize = 5000;
    const ASYNC: usize = 4;
    const COLD: usize = 5;
    let mut rig = Rig::with_host(Box::new(AnswerHost));
    rig.start();
    let mut next = 1u64;
    let mut obj = || {
        next += 1;
        (CONTEXT << RGFX_CONTEXT_OBJ_ID_SHIFT) | next
    };
    let round = |rig: &mut Rig, obj_id: u64| {
        let t0 = Instant::now();
        let seqno = rig.send(object_message(RGFX_CMD_RENDER_PIPELINE_CREATE, obj_id));
        assert_eq!(answer_now(rig), seqno);
        t0.elapsed().as_secs_f64() * 1e6
    };
    // The first, which starts what carries the context's messages out.
    let first = round(&mut rig, obj());
    let warm: Vec<f64> = (0..ROUNDS).map(|_| round(&mut rig, obj())).collect();
    let behind: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            for _ in 0..ASYNC {
                rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj()));
            }
            round(&mut rig, obj())
        })
        .collect();
    let cold: Vec<f64> = (0..COLD)
        .map(|_| {
            std::thread::sleep(Duration::from_millis(1200));
            round(&mut rig, obj())
        })
        .collect();
    println!("a_sync_create_s_round_trip: first {first:.1} us");
    for (name, us) in [("warm", warm), ("behind", behind), ("cold", cold)] {
        let n = us.len();
        let (mean, p50, p99) = stats(us);
        println!(
            "a_sync_create_s_round_trip: {name}: mean {mean:.1} us, p50 {p50:.1}, p99 {p99:.1} \
             ({n} rounds)"
        );
    }
}

/// A context's bursts of `BURST` creates it does not wait for, each followed by a synchronous
/// create, beside `IDLE` other contexts whose lanes have been made and have nothing to do. Prints
/// the time from the burst's first post to the create's answer, with no other context and with
/// them.
#[test]
#[ignore = "a measurement, run by hand"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_burst_beside_idle_lanes() {
    const CONTEXT: u64 = 5;
    const BURST: usize = 200;
    const ROUNDS: usize = 300;
    for idle in [0u64, 75] {
        let mut rig = Rig::with_host(Box::new(AnswerHost));
        rig.start();
        let mut next = 1u64;
        let mut obj = |context: u64| {
            next += 1;
            (context << RGFX_CONTEXT_OBJ_ID_SHIFT) | next
        };
        for other in 0..idle {
            let seqno = rig.send(object_message(
                RGFX_CMD_RENDER_PIPELINE_CREATE,
                obj(CONTEXT + 1 + other),
            ));
            assert_eq!(answer_now(&mut rig), seqno);
        }
        let us: Vec<f64> = (0..ROUNDS)
            .map(|_| {
                let t0 = Instant::now();
                for _ in 0..BURST {
                    rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(CONTEXT)));
                }
                let seqno = rig.send(object_message(
                    RGFX_CMD_RENDER_PIPELINE_CREATE,
                    obj(CONTEXT),
                ));
                assert_eq!(answer_now(&mut rig), seqno);
                t0.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        let (mean, p50, p99) = stats(us);
        println!(
            "a_burst_beside_idle_lanes: {idle} idle lanes: mean {mean:.1} us, p50 {p50:.1}, p99 \
             {p99:.1} ({BURST} creates and a synchronous one, {ROUNDS} rounds)"
        );
    }
}
