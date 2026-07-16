fn main() {
    let a = 5268900186379573i64 as f64;
    let b = -5268900186379173i64 as f64;
    let total = 400i64 as f64;
    let pa = a / total * 100.0;
    let pb = b / total * 100.0;
    println!("{:e}", pa);
    println!("{:e}", pb);
    println!("{}", pa);
    println!("{}", pb);
    println!("bits {:016x} {:016x}", pa.to_bits(), pb.to_bits());
}
