"""
Module to Find CpGs in The read:
Algorithm: 
ord("C") + ord("G") = 138
From left to right, check the sum of every to consecutive numbers, if the number == 138
accumate one to CpG counter
at the end, divide the number by the read length, accumulate 1 to the CpG Tensor at the end of the rounded number.
"""

from MojoFastTrim.stats import Analyser
from MojoFastTrim import FastqRecord
from math import round


struct CGContent(Analyser, Stringable):
    var cg_content: Tensor[DType.int64]

    fn __init__(inout self):
        self.cg_content = Tensor[DType.int64](100)

    fn tally_read(inout self, record: FastqRecord):
        var previous_base: Int8 = 0
        var current_base: Int8 = 0
        var cg_num = 0

        for index in range(1, record.SeqStr.num_elements()):
            previous_base = record.SeqStr[index - 1]
            current_base = record.SeqStr[index]
            if previous_base + current_base == 138:
                cg_num += 1

        let read_cg_content = round(
            cg_num * 100 / record.SeqStr.num_elements()
        ).to_int()
        self.cg_content[read_cg_content] += 1

    fn report(self) -> Tensor[DType.int64]:
        return self.cg_content

    fn __str__(self) -> String:
        return String("\nThe CpG content tensor is: ") + self.cg_content
