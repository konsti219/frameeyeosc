# frameeyeosc

Reading Steam Frame eye-tracking data (gaze and eye openness) and sending it via OSC.

> [!WARNING]
> This is an experimental Proof of Concept and not intended for the average user. If you want proper eye-tracking wait for Babble to release their dedicated Frame software.

Against what was suspected early on, the Steam Frame does also track eye openness (lid position) in addition to gaze.
However, this data is only exposed in an internal shared-memory object (`/dev/shm/eye-server.mmap`), and not via any public APIs.
This small headless program reads data sends it out via OSC as the standard VRCFT parameters, with a configurable prefix.

On the headset, install Rust/Cargo, then build and run normally:

```sh
cargo build --release
./target/release/frameeyeosc --target 127.0.0.1:9000
```

## Command-line arguments

- `--target HOST:PORT` sets the OSC destination (default `127.0.0.1:9000`).
- `--prefix PATH` sets the avatar-parameter prefix (default `/FT`).
- `--raw-lids` sends the raw eye openness values instead of the hand tuned remap.
