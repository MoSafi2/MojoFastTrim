from MojoFastTrim.stats import Analyser
from MojoFastTrim.helpers import write_to_buff
from MojoFastTrim import FastqRecord
from tensor import TensorShape
from collections import Dict, KeyElement


alias MAX_LENGTH = 10_000
alias WIDTH = 5


struct BasepairDistribution(Analyser, Stringable):
    var bp_dist: Tensor[DType.int64]
    # var bp_dist: Dict[StringKey, Tensor[DType.int64]]
    var max_length: Int

    fn __init__(inout self):
        # Hack untill finding a away to grown a multidim tensor in place.
        let shape = TensorShape(WIDTH, MAX_LENGTH)
        self.bp_dist = Tensor[DType.int64](shape)
        self.max_length = 0

    fn tally_read(inout self, record: FastqRecord):
        if record.SeqStr.num_elements() > self.max_length:
            # BUG: Copying multi-dim tensor is not working
            # let t = grow_matrix(self.bp_dist, record.SeqStr.num_elements())
            self.max_length = record.SeqStr.num_elements()

        for i in range(record.SeqStr.num_elements()):
            let index = VariadicList[Int]((record.SeqStr[i] % WIDTH).to_int(), i)
            self.bp_dist[index] += 1

    fn report(self) -> Tensor[DType.int64]:
        # return self.bp_dist

        let final_shape = TensorShape(WIDTH, self.max_length)
        var final_t = Tensor[DType.int64](WIDTH, self.max_length)

        for i in range(WIDTH):
            for j in range(self.max_length):
                let index = VariadicList[Int](i, j)
                final_t[index] = self.bp_dist[index]
        return final_t

    fn __str__(self) -> String:
        return String("\nBase_pair_dist_matrix: ") + self.report()


fn grow_matrix[T: DType](old_tensor: Tensor[T], num_ele: Int) -> Tensor[T]:
    var new_tensor = Tensor[T](WIDTH * num_ele)
    let reshape_e = TensorShape(old_tensor.num_elements())
    var old_reshaped = old_tensor
    try:
        old_reshaped.ireshape(reshape_e)
    except Error:
        print("Error", Error)

    for i in range(old_reshaped.num_elements()):
        new_tensor[i] = old_reshaped[i]

    let new_shape = TensorShape(WIDTH, num_ele)
    try:
        new_tensor.ireshape(new_shape)
    except Error:
        print("Error", Error)

    # for i in range(WIDTH):
    #     for j in range((old_tensor.num_elements() / WIDTH).to_int()):
    #         let index = VariadicList[Int](i, j)
    #         new_tensor[index] = old_tensor[index]
    return new_tensor
