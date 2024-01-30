from collections import Dict, KeyElement
import MojoFastTrim
import time

# TODO: Implemente a rolling DNA hashing algorithm

alias C_seq = ord("C")
alias G_seq = ord("G")
alias A_seq = ord("A")
alias T_seq = ord("T")
alias N_seq = ord("N")


@value
struct DNASequence(KeyElement):
    var s: String

    fn __init__(inout self, owned s: String):
        self.s = s ^

    fn __init__(inout self, s: StringLiteral):
        self.s = String(s)

    fn __hash__(self) -> Int:
        return hash(self.s)

    fn __eq__(self, other: Self) -> Bool:
        return self.s == other.s


# Average 200 ns per record
# Does not deal with N.
fn hash_fun(seq: Tensor[DType.int8]) -> UInt64:
    var hash: UInt64 = 0
    for i in range(seq.num_elements()):
        hash = hash << 2
        hash += (seq[i] & 0x3).to_int()  # Get the lowest two bit
    return hash


fn main() raises:
    var parser = MojoFastTrim.FastqParser("data/102_20.fq")
    var count = 0
    let t1 = time.now()
    let limit = 10
    while True:
        let read = parser.next()
        let h = hash_fun(read.SeqStr)
        count += 1
        print(h)
        if count == limit:
            break

    let t2 = time.now()
    print(t2 - t1)


# Base: 34141299
# With hash: 34358600
