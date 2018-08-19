The codes in the folder are supplementary material for our paper "A Sparse Topic Model for Extracting Aspect-Specific Summaries from Online Reviews." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.

There are three programs in the folder:

1. M-ASUM.py: The proposed M-ASUM sentiment model, where we assign topic for a whole sentence.
2. APSUM_A.py: The proposed aspect-specific sentiment model APSUM without the topic aggregator
3. APSUM_v7.py: This is the final version of the proposed APSUM model (Figure 2(b) of our paper)

For programs 1 and 2 readers can simply run the python file (see the code for details)

The final model is impelmented in APSUM_v7.py, which can be run as follows:

python APSEN_v7.py -d <dataset name> -q <query name>

For other arugments do the following:

python APSUM_v7.py -h

The codes are heavily commented and self explanatory. Feel free to contact vineeth.mohan@technicolor.com if you have any questions.
