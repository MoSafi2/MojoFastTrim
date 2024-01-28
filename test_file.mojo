import time


# Make a function to convert a SIMD value to Binary representation of it
fn main() raises:
    # MAX number of BPs is 16 at a time
    let t1 = time.now()
    let a = SIMD[DType.int8, 16](
        125, 52, 78, 68, 57, 78, 68, 86, 86, 78, 78, 98, 54, 78, 78, 56
    )
    for i in range(15):
        let c = a % 5
        var b: UInt64 = 0
        for i in range(len(a)):
            b = b + (c[i].to_int() * 10**i)
    let t2 = time.now()
    print("time", t2 - t1)
