from MojoFastTrim.stats import Analyser
from MojoFastTrim.helpers import write_to_buff
from MojoFastTrim import FastqRecord
from tensor import TensorShape
from collections import Dict, KeyElement

alias MAX_QUALITY = 100
alias MAX_LENGTH = 10_000
alias OFFSET = 33

# TODO: The encoding of the Fastq file should be predicted and used
# FROM: https://www.biostars.org/p/90845/
# RANGES = {
#     'Sanger': (33, 73),
#     'Solexa': (59, 104),
#     'Illumina-1.3': (64, 104),
#     'Illumina-1.5': (66, 104),
#     'Illumina-1.8': (33, 94)
# }


struct QualityDistribution(Analyser, Stringable):
    var qu_dist: Tensor[DType.int64]
    var max_length: Int
    var max_qu: Int

    fn __init__(inout self):
        # Hack untill finding a away to grown a tensor in place.
        let shape = TensorShape(MAX_QUALITY, MAX_LENGTH)
        self.qu_dist = Tensor[DType.int64](shape)
        self.max_length = 0
        self.max_qu = 0

    fn tally_read(inout self, record: FastqRecord):
        if record.QuStr.num_elements() > self.max_length:
            self.max_length = record.QuStr.num_elements()

        for i in range(record.QuStr.num_elements()):
            let index = VariadicList[Int]((record.QuStr[i] - OFFSET).to_int(), i)
            if record.QuStr[i].to_int() - OFFSET > self.max_qu:
                self.max_qu = record.QuStr[i].to_int() - OFFSET
            self.qu_dist[index] += 1

    fn report(self) -> Tensor[DType.int64]:
        print(self.max_length)
        let final_shape = TensorShape(self.max_qu, self.max_length)
        var final_t = Tensor[DType.int64](final_shape)

        for i in range(self.max_qu):
            for j in range(self.max_length):
                let index = VariadicList[Int](i, j)
                final_t[index] = self.qu_dist[index]
        return final_t

    fn __str__(self) -> String:
        return String("\nQuality_dist_matrix: ") + self.report()
