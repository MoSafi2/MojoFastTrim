from collections import Dict, KeyElement
from MojoFastTrim import FastParser, FastqRecord
from MojoFastTrim.helpers import get_next_line
import time
from base64 import b64encode
from tensor import TensorShape

fn get_fastq_records() raises -> DynamicVector[Tensor[DType.int8]]:
    var vec = DynamicVector[Tensor[DType.int8]](capacity=4)
    let f = open("data/fastq_test.fastq", "r")
    let t = f.read_bytes()

    var offset = 0
    for i in range(4):
        let line = get_next_line(t, offset)
        vec.push_back(line)
        offset += line.num_elements() + 1
    return vec


fn main2() raises:
    
    let valid_vec = get_fastq_records()
    var read = FastqRecord(valid_vec[0], valid_vec[1], valid_vec[2], valid_vec[3])
    var t = Tensor[DType.int64](read.SeqStr.num_elements(), 5)
    
    let t1 = time.now()
    for i in range(read.SeqStr.num_elements()):
        let index = VariadicList[Int](i, (read.SeqStr[i] % 5).to_int())
        t[index] += 1
    let t2 = time.now()
    print(t2-t1)
    print(t)

    for i in range(t.num_elements()):
        print(t[i])
    print(t)


fn main() raises:
    var t = Tensor[DType.int8](20)
    let new_shape = TensorShape(5, 4)
    t.ireshape(new_shape)
    print(t)