from MojoFastTrim.stats import Analyser
from MojoFastTrim.helpers import write_to_buff
from MojoFastTrim import FastqRecord
from tensor import TensorShape
from collections import Dict, KeyElement


@value
struct StringKey(KeyElement):
    var s: String

    fn __init__(inout self, owned s: String):
        self.s = s ^

    fn __init__(inout self, s: StringLiteral):
        self.s = String(s)

    fn __hash__(self) -> Int:
        return hash(self.s)

    fn __eq__(self, other: Self) -> Bool:
        return self.s == other.s


struct BasepairDistribution(Analyser, Stringable):
    var bp_dist: Tensor[DType.int64]
    # var bp_dist: Dict[StringKey, Tensor[DType.int64]]
    var max_shape: Int

    fn __init__(inout self):
        let shape = TensorShape(
            5, 10_000
        )  # Hack untill finding a away to grown a tensor in place.
        self.bp_dist = Tensor[DType.int64](shape)

        # self.bp_dist = Dict[StringKey, Tensor[DType.int64]]()
        # self.bp_dist["t"] = Tensor[DType.int64](shape)
        self.max_shape = 0

    fn tally_read(inout self, record: FastqRecord):
        if record.SeqStr.num_elements() > self.max_shape:
            self.max_shape = record.SeqStr.num_elements()

        for i in range(record.SeqStr.num_elements()):
            let index = VariadicList[Int]((record.SeqStr[i] % 5).to_int(), i)
            self.bp_dist[index] += 1

    fn report(self) -> Tensor[DType.int64]:
        # return self.bp_dist
        let final_shape = TensorShape(5, self.max_shape)
        var final_t = Tensor[DType.int64](5, self.max_shape)

        for i in range(5):
            for j in range(self.max_shape):
                let index = VariadicList[Int](i, j)
                final_t[index] = self.bp_dist[index]
        return final_t

    # BUG: Extremly slow impelementation for incrementing Tensor
    # fn tally_read(inout self, record: FastqRecord):
    #     if record.SeqStr.num_elements() > self.max_shape:
    #         self.max_shape = record.SeqStr.num_elements()

    #     if (
    #         record.SeqStr.num_elements() * 5
    #         > self.bp_dist.find("t").value().num_elements()
    #     ):
    #         let new_tensor = grow_matrix(
    #             self.bp_dist.find("t").value(), record.SeqStr.num_elements()
    #         )
    #         self.bp_dist["t"] = new_tensor

    #     for i in range(record.SeqStr.num_elements()):
    #         let index = VariadicList[Int]((record.SeqStr[i] % 5).to_int(), i)
    #         try:
    #             self.bp_dist["t"][index] += 1
    #         except:
    #             print("failed addition")

    # fn report(self) -> Tensor[DType.int64]:
    #     # return self.bp_dist
    #     return self.bp_dist.find("t").value()

    fn __str__(self) -> String:
        return String("\nBase_pair_dist_matrix: ") + self.report()


fn grow_matrix[
    T: DType,
](old_tensor: Tensor[T], num_ele: Int) -> Tensor[T]:
    let new_shape = TensorShape(5, num_ele)
    var new_tensor = Tensor[T](new_shape)
    for i in range(5):
        for j in range(old_tensor.num_elements() / 5):
            let index = VariadicList[Int](i, j)
            new_tensor[index] = old_tensor[index]
    return new_tensor
