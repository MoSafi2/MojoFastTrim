from MojoFastTrim.stats import Analyser
from MojoFastTrim.helpers import write_to_buff

struct LengthDistribution(Analyser, Stringable):
    var length_vector: Tensor[DType.int64]

    fn __init__(inout self):
        self.length_vector = Tensor[DType.int64](0)

    fn tally_read(inout self, record: FastqRecord):
        if record.SeqStr.num_elements() > self.length_vector.num_elements():
            self.length_vector = grow_tensor(
                self.length_vector, record.SeqStr.num_elements()
            )
        self.length_vector[record.SeqStr.num_elements() - 1] += 1

    @always_inline
    fn length_average(self, num_reads: Int) -> Float64:
        var cum: Int64 = 0
        for i in range(self.length_vector.num_elements()):
            cum += self.length_vector[i] * (i + 1)
        return cum.to_int() / num_reads

    fn report(self) -> Tensor[DType.int64]:
        return self.length_vector

    fn __str__(self) -> String:
        return String("\nLength Distribution: ")+self.length_vector


fn grow_tensor[
    T: DType,
](old_tensor: Tensor[T], num_ele: Int) -> Tensor[T]:
    var new_tensor = Tensor[T](num_ele)
    write_to_buff(old_tensor, new_tensor, 0)

    return new_tensor
