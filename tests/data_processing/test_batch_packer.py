from scaletraining.data_processing.batch_packer import *

def test_group_texts():
    sequences = [
        [[1,2,3],[5,581,23,21],[1,1000000]],
        [[2,3,1],[1,1,1,1,1,1,1,],[1],[5],[1000000]]
    ]

    block_sizes = [2,4,8,16,32,10000]

    for sequence in sequences:
        flat = [t for s in sequence for t in s]
        concat_len = sum(len(s) for s in sequence)
        for bsz in block_sizes:
            result = group_texts({"input_ids": sequence}, block_size=bsz)

            assert isinstance(result, dict)
            assert set(result.keys()) == {"input_ids"}

            blocks = result['input_ids']

            if concat_len < bsz:
                assert blocks == []
                continue
            
            expected_total = (concat_len // bsz) * bsz if bsz > 0 else 0
            expected_num_blocks = expected_total // bsz if bsz > 0 else 0

            assert [x for b in blocks for x in b] == flat[:expected_total]

            assert len(blocks) == expected_num_blocks

            for b in blocks:
                assert len(b) == bsz
                
    print("Group texts passed")

            
    


def main():
    test_group_texts()

if __name__ == "__main__":
    main()