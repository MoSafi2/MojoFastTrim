from MojoFastTrim.stats import Analyser
from MojoFastTrim.helpers import write_to_buff
from MojoFastTrim import FastqRecord
from tensor import TensorShape


struct BasepairDistribution(Analyser):
    var bp_dist: Tensor[DType.int64]

    fn __init__(inout self):
        let shape = TensorShape(1, 5)
        self.bp_dist = Tensor[DType.int64](shape)

    fn tally_read(inout self, record: FastqRecord):
        if record.SeqStr.num_elements() * 5 > self.bp_dist.num_elements():
            var new_tensor = grow_tensor(self.bp_dist, record.SeqStr.num_elements() * 5)
            let new_shape = TensorShape(5, record.SeqStr.num_elements())
            try:
                self.bp_dist = new_tensor.reshape(new_shape)
            except Error:
                print("reshape failed")
                print(Error)

        for i in range(record.SeqStr.num_elements()):
            let index = VariadicList[Int](i, (record.SeqStr[i] % 5).to_int())
            self.bp_dist[index] += 1

    fn report(self) -> Tensor[DType.int64]:
        return self.bp_dist


fn grow_tensor[
    T: DType,
](old_tensor: Tensor[T], num_ele: Int) -> Tensor[T]:
    var new_tensor = Tensor[T](num_ele)
    write_to_buff(old_tensor, new_tensor, 0)
    return new_tensor
