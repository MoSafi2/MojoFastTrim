import time


# Make a function to convert a SIMD value to Binary representation of it
fn main() raises:
    # MAX number of BPs is 16 at a time
    let t1 = time.now()
    let a = SIMD[DType.int8, 16](5, 4, 3, 2, 5, 8, 9, 8, 8, 5, 6, 2, 4, 5, 9, 6)
    var b: UInt64 = 0
    for i in range(len(a)):
        b = b + (a[i].to_int() * 10**i)
    let t2 = time.now()
    print("hash", b)
    print("time", t2 - t1)


fn bin(owned n: Int) -> String:
    if n == 0:
        return "0"
    var binary = String("")
    while n > 0:
        binary = String(n % 2) + binary
        n = n // 2
    return binary
