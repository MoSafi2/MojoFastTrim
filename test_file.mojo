import time
import math

# Make a function to convert a SIMD value to Binary representation of it
fn main() raises:
    # MAX number of BPs is 16 at a time
    let t1 = time.now()
    let a = SIMD[DType.int8, 32](
        125, 52, 78, 68, 57, 78, 68, 86, 86, 78, 78, 98, 54, 78, 78, 56, 125, 52, 78, 68, 57, 78, 68, 86, 86, 78, 78, 98, 54, 78, 78, 56
    )

    print(math.bitcast[DType.int64, 4](a & 0x3))
    let t2 = time.now()
    print("time", t2 - t1)
