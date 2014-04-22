
#Test the content of a SequenceFile that has mapping from int to vector
#If the row id is specified, the content of the specified row will be shown

hadoop jar algebra-1.0-SNAPSHOT-job.jar com.twitter.algebra.matrix.text.TestSequenceFile $*
