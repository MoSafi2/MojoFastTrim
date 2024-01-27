from MojoFastTrim.stats import Analyser
from MojoFastTrim.helpers import write_to_buff
from MojoFastTrim import FastqRecord
from tensor import TensorShape


struct BasepairDistribution(Analyser, Stringable):
    var bp_dist: Tensor[DType.int64]
    var max_shape: Int

    fn __init__(inout self):
        let shape = TensorShape(5, 7000)
        self.bp_dist = Tensor[DType.int64](shape)
        self.max_shape = 0

    fn tally_read(inout self, record: FastqRecord):
        if record.SeqStr.num_elements()> self.max_shape:
            self.max_shape = record.SeqStr.num_elements()

        ## BUG: For some reason the following does not work
        # if record.SeqStr.num_elements() * 5 > self.bp_dist.num_elements():
        #     let new_tensor = grow_tensor(self.bp_dist, record.SeqStr.num_elements())
        #     print("New_Tensor:", new_tensor)
        #     print("BP_list:", self.bp_dist)
        #     self.bp_dist = new_tensor
        #     _ = self.bp_dist
        #     _ = new_tensor

        for i in range(record.SeqStr.num_elements()):
            let index = VariadicList[Int]((record.SeqStr[i] % 5).to_int(), i)
            self.bp_dist[index] += 1

    fn report(self) -> Tensor[DType.int64]:
        let final_shape = TensorShape(5, self.max_shape)
        var final_t = Tensor[DType.int64](5, self.max_shape)
        for i in range(final_t.num_elements()):
            final_t[i] = self.bp_dist[i]
        return final_t
        #return self.bp_dist

    fn __str__(self) -> String:
        return String("\nBase_pair_dist_matrix: ") + self.report()


fn grow_tensor[
    T: DType,
](old_tensor: Tensor[T], num_ele: Int) -> Tensor[T]:
    let new_shape = TensorShape(5, num_ele)
    var new_tensor = Tensor[T](new_shape)
    write_to_buff(old_tensor, new_tensor, 0)
    return new_tensor
