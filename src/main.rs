//! Steam Frame eye bridge using the private eye-server shared-memory ABI (versions 4 and 5).

use clap::Parser;
use memmap2::{MmapMut, MmapOptions};
use rosc::{OscMessage, OscPacket, OscType, encoder};
use std::error::Error;
use std::fs::OpenOptions;
use std::io;
use std::mem::{align_of, offset_of, size_of};
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::ptr;
use std::time::Duration;

const SOURCE: &str = "/dev/shm/eye-server.mmap";
const TIMEOUT: Duration = Duration::from_secs(1);

fn abi(version: u32) -> Option<(usize, usize)> {
    match version {
        4 => Some((0x4f21a, 0x152)), // 0.5.0, build 20260921
        5 => Some((0x4f21f, 0x157)), // build 20260930
        _ => None,
    }
}

#[repr(C)]
struct EyeServerHeader {
    version: u32,
    initialized: u32,
    // The target glibc mutex slot is 48 bytes; host libc may define a smaller type.
    metadata_mutex: [u8; 0x30],
    sequence: u32,
    metadata_requested: u32,
}

// The record is packed, so its timestamp and vectors are not naturally aligned.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct EyeDataMmap {
    producer_state: u32,
    sample_flag: u8,
    sample_time: f64,
    // Left, right; after stereo fusion.
    gaze_direction: [[f32; 3]; 2],
    gaze_covariance_diag: [[f32; 3]; 2],
    // Head-relative, -Z forward, in metres.
    fixation_point: [f32; 3],
    pre_fusion_gaze: [[f32; 3]; 2],
    pre_fusion_cov_diag: [[f32; 3]; 2],
    openness: [f32; 2],
    estimate_extra: [f32; 8],
    reserved: [u8; 0xe1b],
}

const _: () = {
    assert!(offset_of!(EyeServerHeader, metadata_mutex) == 0x08);
    assert!(offset_of!(EyeServerHeader, sequence) == 0x38);
    assert!(offset_of!(EyeServerHeader, metadata_requested) == 0x3c);
    assert!(offset_of!(EyeDataMmap, sample_time) == 0x05);
    assert!(offset_of!(EyeDataMmap, gaze_direction) == 0x0d);
    assert!(offset_of!(EyeDataMmap, gaze_covariance_diag) == 0x25);
    assert!(offset_of!(EyeDataMmap, fixation_point) == 0x3d);
    assert!(offset_of!(EyeDataMmap, pre_fusion_gaze) == 0x49);
    assert!(offset_of!(EyeDataMmap, pre_fusion_cov_diag) == 0x61);
    assert!(offset_of!(EyeDataMmap, openness) == 0x79);
    assert!(size_of::<EyeDataMmap>() == 0xebc);
    assert!(size_of::<libc::pthread_mutex_t>() <= 0x30);
    assert!(8 % align_of::<libc::pthread_mutex_t>() == 0);
};

#[derive(Parser)]
#[command(about = "Send Steam Frame eye tracking from shared memory over OSC")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:9000")]
    target: String,
    #[arg(long, default_value = "/FT")]
    prefix: String,
    /// Send the raw eye openness values instead of the hand tuned remap.
    #[arg(long)]
    raw_lids: bool,
}

struct EyeSource {
    map: MmapMut,
    record_offset: usize,
}

struct MutexGuard(*mut libc::pthread_mutex_t);

impl Drop for MutexGuard {
    fn drop(&mut self) {
        unsafe { libc::pthread_mutex_unlock(self.0) };
    }
}

struct EyeData {
    sample_time: f64,
    gaze: [[f32; 3]; 2],
    fixation_point: [f32; 3],
    openness: [f32; 2],
}

// Valves eyelid output is kinda bad and asymmetric, so do a shitty hand tuned remap per eye
const OPENNESS_CLOSED: [f32; 2] = [0.26, 0.29];
const OPENNESS_OPEN: [f32; 2] = [0.70, 0.75];
const OPENNESS_WIDE: [f32; 2] = [0.88, 0.83];
const LID_NEUTRAL: f32 = 0.75;
const LID_SMOOTHING: f32 = 0.4;

fn remap_lid(eye: usize, value: f32) -> f32 {
    let (closed, open, wide) = (OPENNESS_CLOSED[eye], OPENNESS_OPEN[eye], OPENNESS_WIDE[eye]);
    let lid = if value < open {
        (value - closed) / (open - closed) * LID_NEUTRAL
    } else {
        LID_NEUTRAL + (value - open) / (wide - open) * (1.0 - LID_NEUTRAL)
    };
    lid.clamp(0.0, 1.0)
}

impl EyeSource {
    fn open() -> Result<Self, Box<dyn Error>> {
        let file = OpenOptions::new().read(true).write(true).open(SOURCE)?;
        let len = file.metadata()?.len() as usize;
        if len < size_of::<EyeServerHeader>() {
            return Err(format!("{SOURCE}: shared memory is too small").into());
        }
        let map = unsafe { MmapOptions::new().len(len).map_mut(&file)? };
        let header: *const EyeServerHeader = map.as_ptr().cast();
        let version = u32::from_le(unsafe { ptr::read_volatile(&raw const (*header).version) });
        let (size, record_offset) = abi(version).ok_or_else(|| {
            format!("Unsupported eye shared-memory version {version}. Only  4 and 5 are supported")
        })?;
        if len < size {
            return Err(format!("{SOURCE}: {len} bytes, version {version} needs {size}").into());
        }
        if u32::from_le(unsafe { ptr::read_volatile(&raw const (*header).initialized) }) != 1 {
            return Err("Eye shared memory is not initialized".into());
        }
        eprintln!("Eye shared memory version {version}");
        Ok(Self { map, record_offset })
    }

    fn layout(&self) -> *const EyeServerHeader {
        self.map.as_ptr().cast()
    }

    fn layout_mut(&mut self) -> *mut EyeServerHeader {
        self.map.as_mut_ptr().cast()
    }

    fn lock(&mut self) -> io::Result<MutexGuard> {
        let mutex = unsafe { (&raw mut (*self.layout_mut()).metadata_mutex).cast() };
        let code = unsafe { libc::pthread_mutex_lock(mutex) };
        if code == libc::EOWNERDEAD {
            let result = unsafe { libc::pthread_mutex_consistent(mutex) };
            if result != 0 {
                unsafe { libc::pthread_mutex_unlock(mutex) };
                return Err(io::Error::from_raw_os_error(result));
            }
        } else if code != 0 {
            return Err(io::Error::from_raw_os_error(code));
        }
        Ok(MutexGuard(mutex))
    }

    fn next(&mut self, timeout: Duration) -> io::Result<Option<EyeData>> {
        let guard = self.lock()?;
        let sequence_ptr = unsafe { &raw const (*self.layout()).sequence };
        let sequence = unsafe { ptr::read_volatile(sequence_ptr) };
        let request_ptr = unsafe { &raw mut (*self.layout_mut()).metadata_requested };
        unsafe { ptr::write_volatile(request_ptr, 1) };
        drop(guard);

        let timespec = libc::timespec {
            tv_sec: timeout.as_secs() as libc::time_t,
            tv_nsec: timeout.subsec_nanos() as libc::c_long,
        };
        let result = unsafe {
            libc::syscall(
                libc::SYS_futex,
                sequence_ptr,
                libc::FUTEX_WAIT,
                sequence,
                &timespec as *const libc::timespec,
            )
        };
        if result == -1 {
            let error = io::Error::last_os_error();
            if !matches!(
                error.raw_os_error(),
                Some(libc::EAGAIN | libc::EINTR | libc::ETIMEDOUT)
            ) {
                return Err(error);
            }
        }

        let guard = self.lock()?;
        let data = if unsafe { ptr::read_volatile(sequence_ptr) } != sequence {
            let record_ptr = unsafe { self.map.as_ptr().add(self.record_offset) };
            let record = unsafe { ptr::read_unaligned(record_ptr.cast::<EyeDataMmap>()) };
            if record.producer_state == 1 {
                Some(EyeData {
                    sample_time: record.sample_time,
                    gaze: record.gaze_direction,
                    fixation_point: record.fixation_point,
                    openness: record.openness,
                })
            } else {
                None
            }
        } else {
            None
        };
        drop(guard);
        Ok(data)
    }
}

fn send(socket: &UdpSocket, addr: String, args: Vec<OscType>) -> Result<(), Box<dyn Error>> {
    let packet = OscPacket::Message(OscMessage { addr, args });
    socket.send(&encoder::encode(&packet)?)?;
    Ok(())
}

// Like Steam Link's OSC sender, ±45° maps to ±1; +Y is up (VRCFT convention, unverified on hardware).
fn gaze_angles([x, y, z]: [f32; 3]) -> [f32; 2] {
    let scale = 4.0 / std::f32::consts::PI;
    [
        (x.atan2(-z) * scale).clamp(-1.0, 1.0),
        (y.atan2(-z) * scale).clamp(-1.0, 1.0),
    ]
}

fn send_eye_data(socket: &UdpSocket, args: &Args, data: EyeData) -> Result<(), Box<dyn Error>> {
    let prefix = format!("/avatar/parameters{}", args.prefix.trim_end_matches('/'));
    send(
        socket,
        format!("{prefix}/EyeTrackingActive"),
        vec![OscType::Bool(true)],
    )?;
    let [left_x, left_y] = gaze_angles(data.gaze[0]);
    let [right_x, right_y] = gaze_angles(data.gaze[1]);
    let [x, y] = gaze_angles(data.fixation_point);
    for (suffix, value) in [
        ("EyeLeftX", left_x),
        ("EyeLeftY", left_y),
        ("EyeRightX", right_x),
        ("EyeRightY", right_y),
        ("EyeLidLeft", data.openness[0]),
        ("EyeLidRight", data.openness[1]),
        ("EyeX", x),
        ("EyeY", y),
    ] {
        send(
            socket,
            format!("{prefix}/v2/{suffix}"),
            vec![OscType::Float(value)],
        )?;
    }
    // Native eye look
    let closed_amount =
        1.0 - ((data.openness[0] + data.openness[1]) / 2.0 / LID_NEUTRAL).clamp(0.0, 1.0);
    send(
        socket,
        format!("/tracking/eye/EyesClosedAmount"),
        vec![OscType::Float(closed_amount)],
    )?;
    let degree_multiplier = 180.0 / 4.0;
    send(
        socket,
        format!("/tracking/eye/LeftRightPitchYaw"),
        vec![
            OscType::Float(-left_y * degree_multiplier),
            OscType::Float(left_x * degree_multiplier),
            OscType::Float(-right_y * degree_multiplier),
            OscType::Float(right_x * degree_multiplier),
        ],
    )?;
    Ok(())
}

fn send_inactive(socket: &UdpSocket, prefix: &str) -> Result<(), Box<dyn Error>> {
    send(
        socket,
        format!(
            "/avatar/parameters{}/EyeTrackingActive",
            prefix.trim_end_matches('/')
        ),
        vec![OscType::Bool(false)],
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if !args.prefix.starts_with('/') || args.prefix.trim_matches('/').is_empty() {
        return Err("--prefix must be a nonempty OSC path starting with /".into());
    }
    let target: SocketAddr = args
        .target
        .to_socket_addrs()?
        .next()
        .ok_or("--target did not resolve to an address")?;
    let socket = UdpSocket::bind(if target.is_ipv4() {
        "0.0.0.0:0"
    } else {
        "[::]:0"
    })?;
    socket.connect(target)?;
    let mut source = EyeSource::open()?;
    eprintln!("Reading {SOURCE} and sending OSC to {target}");
    let mut active = false;
    let mut lids = [LID_NEUTRAL; 2];
    loop {
        match source.next(TIMEOUT)? {
            Some(data)
                if data.sample_time.is_finite()
                    && data.gaze.iter().flatten().all(|value| value.is_finite())
                    && data.fixation_point.iter().all(|value| value.is_finite())
                    && data.openness.iter().all(|value| value.is_finite()) =>
            {
                for eye in 0..2 {
                    let lid = if args.raw_lids {
                        data.openness[eye].clamp(0.0, 1.0)
                    } else {
                        remap_lid(eye, data.openness[eye])
                    };
                    lids[eye] += (lid - lids[eye]) * LID_SMOOTHING;
                }
                send_eye_data(
                    &socket,
                    &args,
                    EyeData {
                        openness: lids,
                        ..data
                    },
                )?;
                active = true;
            }
            _ if active => {
                send_inactive(&socket, &args.prefix)?;
                active = false;
            }
            _ => {}
        }
    }
}
