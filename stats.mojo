"""This module should hold aggregate statistics about all the record which have been queried by the Parser, regardless of the caller function. """

from MojoFastTrim import FastqRecord, RecordCoord
from collections import Dict
from tensor import Tensor, TensorShape
from MojoFastTrim.helpers import write_to_buff
from math import max

alias MAX_COUNTS = 1_000_000


trait Analyser:
    fn tally_read(inout self, record: FastqRecord):
        ...

    fn report(self) -> Tensor[DType.int64]:
        ...


struct Stats(Stringable):
    var num_reads: Int64
    var total_bases: Int64
    var unique_counts: Int

    fn __init__(inout self):
        self.num_reads = 0
        self.total_bases = 0
        # self.sequences = Dict[FastqRecord, Int]()
        self.unique_counts = 0

    # Consider using Internal function for each type to get this, there is no need to know the impelemtnation of each type, this can get Ugly if you want to Add BAM, SAM .. etc.
    @always_inline
    fn tally(inout self, record: FastqRecord):
        self.num_reads += 1
        self.total_bases += record.SeqStr.num_elements()



    fn __str__(self) -> String:
        return (
            String("Number of Reads: ")
            + self.num_reads
            + ". \n"
            + "Number of bases: "
            + self.total_bases
            + ".\n"
            + "Number of Unique reads: "
        )


