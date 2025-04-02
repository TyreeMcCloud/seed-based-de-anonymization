#!/bin/bash
python seed_based.py -nw 32 -g1 data/seed_G1.edgelist -g2 data/seed_G2.edgelist -sm data/seed_mapping_test.txt -out Solution3.txt
python seed_based.py -nw 32 -g1 test/test_g1.edgelist -g2 test/test_g2.edgelist -sm test/test_mapping.txt -out Solution3.txt
#python seed_free_main.py -nw 32 -g1 seedfree/G1.edgelist -g2 seedfree/G2.edgelist -out seedfree/seed_free_result.txt