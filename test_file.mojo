fn main():
    var t: UInt64 = 0xFFFFFFFFFFFFFFFF
    print(((t & 0x3FFFFFFFFFFFFFFF) << 2) + 0)