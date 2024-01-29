"""Idea from FalcoQC, Imeplement your own hash function for Basebairs maybe usign Uptobit?"""
# https://github.com/JohnLonginotto/ACGTrie/blob/master/docs/UP2BIT.md

from collections import Dict, KeyElement
from MojoFastTrim import FastqRecord
from MojoFastTrim.stats import Analyser

alias MAX_READS = 100_000


struct DupReader(Analyser, Stringable):
    var unique_dict: Dict[FastqRecord, Int16]
    var unique_reads: Int

    fn __init__(inout self):
        self.unique_dict = Dict[FastqRecord, Int16]()
        self.unique_reads = 0

    fn tally_read(inout self, record: FastqRecord):
        if self.unique_dict.__contains__(record):
            try:
                self.unique_dict[record] += 1
                return
            except:
                return

        if self.unique_reads < MAX_READS:
            self.unique_dict[record] = 1
            self.unique_reads += 1
        else:
            pass

    fn report(self) -> Tensor[DType.int64]:
        var report = Tensor[DType.int64](1)
        report[0] = len(self.unique_dict)
        return report

    fn __str__(self) -> String:
        return String("\nNumber of duplicated reads is") + self.report()
