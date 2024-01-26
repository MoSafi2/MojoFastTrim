"""This module should hold aggregate statistics about all the record which have been queried by the Parser, regardless of the caller function. """

from MojoFastTrim import FastqRecord, RecordCoord
from collections import Dict
from tensor import Tensor, TensorShape
from MojoFastTrim.helpers import write_to_buff
from math import max

alias MAX_COUNTS = 1_000_000


struct Stats(Stringable):
    var num_reads: Int64
    var total_bases: Int64
    var sequences: Dict[FastqRecord, Int]
    var unique_counts: Int
    var length_vector: Tensor[DType.int64]

    fn __init__(inout self):
        self.num_reads = 0
        self.total_bases = 0
        self.sequences = Dict[FastqRecord, Int]()
        self.unique_counts = 0
        self.length_vector = Tensor[DType.int64](1)

    @always_inline
    fn tally(inout self, record: FastqRecord):
        self.num_reads += 1
        self.total_bases += record.SeqStr.num_elements()
        
        #Length Distribution
        self._check_grow_tensor(self.length_vector, record.SeqStr.num_elements())
        self.length_vector[record.SeqStr.num_elements() - 1] += 1

        # if self.num_reads > MAX_COUNTS:
        #     return

        # if self.sequences.find(record):
        #     try:
        #         self.sequences[record] += 1
        #     except:
        #         pass
        # else:
        #     self.sequences[record] = 1
        #     self.unique_counts += 1

    @always_inline
    fn tally(inout self, record: RecordCoord):
        self.num_reads += 1
        self.total_bases += record.seq_len().to_int()

    fn length_average(self) -> Float64:
        var cum: Int64 = 0
        for i in range(self.length_vector.num_elements()):
            cum += self.length_vector[i] * (i + 1)
        return cum.to_int() / self.num_reads.to_int()

    fn _check_grow_tensor[T: DType](self, inout old_tensor: Tensor[T], ele: Int):
        if old_tensor.num_elements() < ele:
            var new_tensor = Tensor[T](ele)
            write_to_buff(old_tensor, new_tensor, 0)
            old_tensor = new_tensor

    fn __str__(self) -> String:
        return (
            String("Number of Reads: ")
            + self.num_reads
            + ". \n"
            + "Number of bases: "
            + self.total_bases
            + ".\n"
            + "Number of Unique reads: "
            + len(self.sequences)
            + "\nAverage Sequence length:"
            + self.length_average()
        )
