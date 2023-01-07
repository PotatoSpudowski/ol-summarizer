from infer import get_summary

flush = get_summary("flush")

test_input = "In recent years, training ever larger language models has become the norm. While the issues of those models' not being released for further study is frequently discussed, the hidden knowledge about how to train such models rarely gets any attention. This article aims to change this by shedding some light on the technology and engineering behind training such models both in terms of hardware and software on the example of the 176B parameter language model BLOOM."

summary = get_summary(test_input)

print(summary)