use flate2::read::MultiGzDecoder;
use std::io::Read;

fn main() {
    let data = std::fs::read("../concat.pb.gz").unwrap();
    let mut decoder = MultiGzDecoder::new(&data[..]);
    let mut payload = Vec::new();
    decoder.read_to_end(&mut payload).unwrap();
    let hex: String = payload.iter().map(|b| format!("{b:02x}")).collect();
    println!("{hex}");
}
