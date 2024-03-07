USE THIS COMMAND IN TERMINAL TO EXTRACT DUPLICATES FROM THE OUTPUT.TSV FILE
    #  awk '!a[$2]++ && !b[$3]++' /output.tsv > /duplicates_output.tsv 
    # It will counτ the first row as header, so before running this command, please add a header at each column (named: MGYS (col A) and Secondary Accession (col B))
    # Do not skip row 1, it might have duplicates.  