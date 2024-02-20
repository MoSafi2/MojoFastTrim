from MojoFastTrim.helpers import slice_tensor, write_to_buff
from memory.unsafe import DTypePointer
from tensor import Tensor
from base64 import b64encode
from collections import KeyElement
from MojoFastTrim.CONSTS import read_header, new_line, quality_header, USE_SIMD
from MojoFastTrim import fnv1a32, fnv1a64
from math import min
import time


@value
struct FastqRecord(CollectionElement, Sized, Stringable, KeyElement):
    """Struct that represent a single FastaQ record."""

    var SeqHeader: Tensor[DType.int8]
    var SeqStr: Tensor[DType.int8]
    var QuHeader: Tensor[DType.int8]
    var QuStr: Tensor[DType.int8]
    var total_length: Int
    var hash: Int

    fn __init__(
        inout self,
        SH: Tensor[DType.int8],
        SS: Tensor[DType.int8],
        QH: Tensor[DType.int8],
        QS: Tensor[DType.int8],
    ) raises -> None:
        if SH[0] != read_header:
            print(SH)
            raise Error("Sequence Header is corrput")

        if QH[0] != quality_header:
            print(QH)
            raise Error("Quality Header is corrput")

        if SS.num_elements() != QS.num_elements():
            print(
                "SeqStr_length:",
                SS.num_elements(),
                "QualityStr_Length:",
                QS.num_elements(),
            )
            raise Error("Corrput Lengths")

        self.SeqHeader = SH
        self.QuHeader = QH

        if self.QuHeader.num_elements() > 1:
            if self.QuHeader.num_elements() != self.SeqHeader.num_elements():
                print(QH)
                raise Error("Quality Header is corrupt")

        self.SeqStr = SS
        self.QuStr = QS

        self.total_length = (
            SH.num_elements()
            + SS.num_elements()
            + QH.num_elements()
            + QS.num_elements()
            + 4  # Addition of 4 \n again
        )

        self.hash = -1

    fn get_seq(self) -> String:
        var t = self.SeqStr
        return String(t._steal_ptr(), t.num_elements())

    @always_inline
    fn wirte_record(self) -> Tensor[DType.int8]:
        return self.__concat_record()

    @always_inline
    fn _empty_record(inout self):
        let empty = Tensor[DType.int8](0)
        self.SeqStr = empty
        self.SeqHeader = empty
        self.QuStr = empty
        self.QuHeader = empty
        self.total_length = 0

    @always_inline
    fn __concat_record(self) -> Tensor[DType.int8]:
        if self.total_length == 0:
            return Tensor[DType.int8](0)

        var offset = 0
        var t = Tensor[DType.int8](self.total_length)

        write_to_buff(self.SeqHeader, t, offset)
        offset = offset + self.SeqHeader.num_elements() + 1
        t[offset - 1] = new_line

        write_to_buff(self.SeqStr, t, offset)
        offset = offset + self.SeqStr.num_elements() + 1
        t[offset - 1] = new_line

        write_to_buff(self.QuHeader, t, offset)
        offset = offset + self.QuHeader.num_elements() + 1
        t[offset - 1] = new_line

        write_to_buff(self.QuStr, t, offset)
        offset = offset + self.QuStr.num_elements() + 1
        t[offset - 1] = new_line

        return t

    @always_inline
    fn __str__(self) -> String:
        if self.total_length == 0:
            return ""
        var concat = self.__concat_record()
        return String(concat._steal_ptr(), self.total_length)

    @always_inline
    fn __len__(self) -> Int:
        return self.SeqStr.num_elements()

    # Consider changing hash function to another performant one.
    # Hashing the first 48 Neuclotides from each read, average 380 ns, 20x faster than the internal hash function.

    # # 10% Faster in execution for some reason
    @always_inline
    fn __hash__(self) -> Int:
        var hash = SIMD[DType.uint64, 4]()
        var index = 0
        for i in range(3):
            let a = self.SeqStr.simd_load[16](16 * i)
            let b = a % 5
            for j in range(len(b)):
                hash[i] += b[j].to_int() * 10**j
        let final_hash = hash.reduce_add().to_int()
        return final_hash

    fn set_hash(inout self):
        self.hash = self.__hash__()

    # Can be used as a rotating hash, Should be used to adatpers also?
    @always_inline
    fn init_hash(self) -> Int:
        var hash: Int = 1
        let ele = self.SeqStr.simd_load[32](0) & 0x03
        for i in range(31):
            hash = hash << 2
            hash += ele[i].to_int()
        return hash

    # This still very inefficient.
    @always_inline
    fn rot_hash(self, start: Int) -> Int:
        if start == 0:
            return self.hash
        elif start < 16:
            var hash = self.hash
            for i in range(1, start):
                hash = (hash & 0x3FFFFFFFFFFFFFFF) << 2
                hash += self.SeqStr[i + 31].to_int()
            return hash
        else:
            var hash: Int = 1
            let ele = self.SeqStr.simd_load[32](start) & 0x03
            for i in range(31):
                hash = hash << 2
                hash += ele[i].to_int()
            return hash

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.__hash__() == other.__hash__()


fn main() raises:
    var l1 = Tensor[DType.int8](100)
    let l2 = Tensor[DType.int8](100)
    var l3 = Tensor[DType.int8](100)
    let l4 = Tensor[DType.int8](100)
    l1[0] = 64
    l3[0] = 43

    let read = FastqRecord(l1, l2, l3, l4)
    print(read.rot_hash(0))
